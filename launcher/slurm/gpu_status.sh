#!/bin/bash
# GPU Partition Status Report
# Generates a summary of GPU availability by type and node status

set -e

PARTITION="${1:-gpu}"

# Collect node data
TMPFILE=$(mktemp)
trap "rm -f $TMPFILE" EXIT

sinfo -p "$PARTITION" -N -h -o "%N" | xargs -I{} sh -c 'scontrol show node {} 2>/dev/null' | \
    grep -E "NodeName=|State=|Gres=|AllocTRES=" | paste - - - - > "$TMPFILE"

python3 << EOF
import re
from collections import defaultdict

data = defaultdict(lambda: defaultdict(lambda: {'nodes': 0, 'total': 0, 'used': 0, 'avail': 0}))
available_nodes = []

with open("$TMPFILE", 'r') as f:
    for line in f:
        node_match = re.search(r'NodeName=(\S+)', line)
        state_match = re.search(r'State=([A-Z_+]+)', line)
        gres_match = re.search(r'Gres=gpu:([^:]+):(\d+)', line)
        used_match = re.search(r'gres/gpu=(\d+)', line)

        if node_match and state_match and gres_match:
            node = node_match.group(1)
            state = state_match.group(1)
            gtype = gres_match.group(1).upper()
            gtotal = int(gres_match.group(2))
            gused = int(used_match.group(1)) if used_match else 0
            avail = gtotal - gused

            # Categorize state
            if 'DOWN' in state or 'DRAIN' in state:
                scat = 'DOWN/DRAIN'
            elif 'IDLE' in state:
                scat = 'IDLE'
            elif 'ALLOCATED' in state:
                scat = 'ALLOCATED'
            else:
                scat = 'MIXED'

            data[gtype][scat]['nodes'] += 1
            data[gtype][scat]['total'] += gtotal
            data[gtype][scat]['used'] += gused
            data[gtype][scat]['avail'] += avail

            # Track nodes with available GPUs (exclude DOWN/DRAIN)
            if avail > 0 and scat not in ['DOWN/DRAIN']:
                available_nodes.append({
                    'node': node,
                    'gtype': gtype,
                    'total': gtotal,
                    'used': gused,
                    'avail': avail,
                    'state': scat
                })

# Print header
print("=" * 60)
print("         GPU PARTITION STATUS BY GPU TYPE")
print("=" * 60)
print()

grand = {'nodes': 0, 'total': 0, 'used': 0, 'avail': 0}

# Sort GPU types for consistent output
for gtype in sorted(data.keys()):
    type_total = sum(d['total'] for d in data[gtype].values())
    type_nodes = sum(d['nodes'] for d in data[gtype].values())
    type_used = sum(d['used'] for d in data[gtype].values())
    type_avail = sum(d['avail'] for scat, d in data[gtype].items() if scat != 'DOWN/DRAIN')

    print(f">>> {gtype} ({type_nodes} nodes, {type_total} total GPUs)")
    print(f"    {'State':<12} {'Nodes':>6} {'Total':>6} {'Used':>6} {'Avail':>6}")
    print(f"    {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for scat in ['IDLE', 'MIXED', 'ALLOCATED', 'DOWN/DRAIN']:
        if scat in data[gtype]:
            d = data[gtype][scat]
            print(f"    {scat:<12} {d['nodes']:>6} {d['total']:>6} {d['used']:>6} {d['avail']:>6}")

    print(f"    {'SUBTOTAL':<12} {type_nodes:>6} {type_total:>6} {type_used:>6} {type_avail:>6}")
    print()

    grand['nodes'] += type_nodes
    grand['total'] += type_total
    grand['used'] += type_used
    grand['avail'] += type_avail

print("=" * 60)
print(f"GRAND TOTAL: {grand['nodes']} nodes, {grand['total']} GPUs ({grand['used']} used, {grand['avail']} available)")
print("=" * 60)

# Print available nodes table sorted by GPU type and available count
print()
print("=" * 70)
print("         NODES WITH AVAILABLE GPUs (sorted by GPU Type, Available)")
print("=" * 70)
print()

if available_nodes:
    # Sort by GPU type (ascending), then by available GPUs (descending)
    available_nodes.sort(key=lambda x: (x['gtype'], -x['avail'], x['node']))

    print(f"{'Node':<15} {'GPU Type':<8} {'Total':>6} {'Used':>6} {'Avail':>6} {'State':<12}")
    print(f"{'-'*15} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*12}")

    for n in available_nodes:
        print(f"{n['node']:<15} {n['gtype']:<8} {n['total']:>6} {n['used']:>6} {n['avail']:>6} {n['state']:<12}")

    print()
    print(f"Total: {len(available_nodes)} nodes with {sum(n['avail'] for n in available_nodes)} GPUs available")
else:
    print("No nodes with available GPUs found.")

print()
EOF
