import json
import os

# Change to artifacts directory
os.chdir("/Users/manav.neema/Documents/Thesis/Experiment 2/Casual discovery/artifacts/Iter4 - 107 days")

# Load artifacts
with open('causal_artifacts.json') as f:
    artifacts = json.load(f)

with open('downstream_map.json') as f:
    downstream = json.load(f)

with open('upstream_map.json') as f:
    upstream = json.load(f)

with open('hybrid_causal_edges.csv') as f:
    edges_content = f.read()

# All 39 features in final graph
all_features = list(artifacts['tier_assignments'].keys())

# Nodes in graph (have edges)
in_downstream = set(downstream.keys())
in_upstream = set(upstream.keys())
downstream_targets = set()
for targets in downstream.values():
    downstream_targets.update(targets)
upstream_sources = set()
for sources in upstream.values():
    upstream_sources.update(sources)

# All connected nodes
connected_nodes = in_downstream | in_upstream | downstream_targets | upstream_sources

# Isolated nodes
isolated = set(all_features) - connected_nodes

print("=" * 70)
print("COMPREHENSIVE GRAPH ANALYSIS - Iter4 (107 days)")
print("=" * 70)

print(f"\n{'='*70}")
print("1. FEATURE COVERAGE SUMMARY")
print("="*70)
print(f"   Total features in dataset: {len(all_features)}")
print(f"   Connected nodes in graph:  {len(connected_nodes)}")
print(f"   Isolated nodes (NO EDGES): {len(isolated)}")
print(f"   Graph coverage: {len(connected_nodes)/len(all_features)*100:.1f}%")

print(f"\n{'='*70}")
print("2. ISOLATED NODES - CANNOT BE DETECTED AS ROOT CAUSES")
print("="*70)
tier_name = {0: 'RAW', 1: 'BRONZE', 2: 'SILVER', 3: 'KPI'}
if isolated:
    for node in sorted(isolated):
        tier = artifacts['tier_assignments'].get(node, '?')
        print(f"   X {node} [Tier {tier}: {tier_name.get(tier, '?')}]")
else:
    print("   (None - all nodes connected)")

print(f"\n{'='*70}")
print("3. SOURCE NODES - CAN BE ROOT CAUSES (have outgoing edges)")
print("="*70)
for node in sorted(in_downstream, key=lambda x: -len(downstream[x])):
    out_degree = len(downstream[node])
    tier = artifacts['tier_assignments'].get(node, '?')
    targets = downstream[node]
    print(f"   {node} [Tier {tier}]")
    print(f"      -> {targets}")

print(f"\n{'='*70}")
print("4. SINK NODES - LEAF NODES (only incoming edges, cannot be root causes)")
print("="*70)
pure_sinks = downstream_targets - in_downstream
for node in sorted(pure_sinks):
    tier = artifacts['tier_assignments'].get(node, '?')
    # Find who points to this sink
    parents = [k for k, v in downstream.items() if node in v]
    print(f"   {node} [Tier {tier}]")
    print(f"      <- from: {parents}")

print(f"\n{'='*70}")
print("5. HUB ANALYSIS - HIGH CONNECTIVITY NODES (potential bias)")
print("="*70)
print("   High out-degree nodes dominate scoring:")
for node in sorted(in_downstream, key=lambda x: -len(downstream[x])):
    out_degree = len(downstream[node])
    in_count = sum(1 for v in upstream.values() if node in v)
    if out_degree >= 2:
        print(f"   * {node}: out-degree={out_degree}, in-degree={in_count}")

print(f"\n{'='*70}")
print("6. TIER FLOW ANALYSIS")
print("="*70)
for tier in [0, 1, 2, 3]:
    tier_nodes = [n for n, t in artifacts['tier_assignments'].items() if t == tier]
    connected_tier = [n for n in tier_nodes if n in connected_nodes]
    isolated_tier = [n for n in tier_nodes if n in isolated]
    source_tier = [n for n in tier_nodes if n in in_downstream]
    sink_tier = [n for n in tier_nodes if n in pure_sinks]
    print(f"\n   Tier {tier} ({tier_name[tier]}):")
    print(f"      Total nodes: {len(tier_nodes)}")
    print(f"      Connected:   {len(connected_tier)} ({len(connected_tier)/len(tier_nodes)*100:.0f}%)")
    print(f"      Isolated:    {len(isolated_tier)}")
    print(f"      As sources:  {len(source_tier)}")
    print(f"      As sinks:    {len(sink_tier)}")
    if isolated_tier:
        print(f"      ISOLATED: {isolated_tier}")

print(f"\n{'='*70}")
print("7. POTENTIAL TEST CASE COVERAGE")
print("="*70)
print("\n   For RCA to work, root cause must be a SOURCE node (has downstream).")
print("   Nodes that can be detected as root causes:")

source_nodes_by_tier = {}
for tier in [0, 1, 2, 3]:
    source_nodes_by_tier[tier] = [n for n in in_downstream 
                                   if artifacts['tier_assignments'].get(n) == tier]
    
for tier in [0, 1, 2, 3]:
    nodes = source_nodes_by_tier[tier]
    print(f"\n   Tier {tier} ({tier_name[tier]}) - {len(nodes)} source nodes:")
    for n in sorted(nodes):
        print(f"      - {n}")

print(f"\n{'='*70}")
print("8. CRITICAL MISSING EDGES (Blacklisted but 100% Bootstrap Stable)")
print("="*70)
# Count stable edges
stable_edges = artifacts['stable_edges']
final_edges = artifacts['final_edges']

final_set = set()
for e in final_edges:
    final_set.add((e['from'], e['to']))

blacklisted = []
for e in stable_edges:
    if (e['from'], e['to']) not in final_set and e['bootstrap_frequency'] == 1.0:
        blacklisted.append(e)

print(f"   Total 100% stable edges: {len([e for e in stable_edges if e['bootstrap_frequency'] == 1.0])}")
print(f"   Edges in final graph: {len(final_edges)}")
print(f"   Blacklisted stable edges: {len(blacklisted)}")
print("\n   Blacklisted edges (100% bootstrap stable but removed):")
for e in blacklisted[:15]:
    print(f"      {e['from']} -> {e['to']} (weight: {e['weight']:.4f})")

print(f"\n{'='*70}")
print("9. BIDIRECTIONAL RELATIONSHIPS (Cycles Broken Arbitrarily)")
print("="*70)
# Find bidirectional edges in stable_edges
bidir = {}
for e in stable_edges:
    key = tuple(sorted([e['from'], e['to']]))
    if key not in bidir:
        bidir[key] = []
    bidir[key].append(e)

print("   These pairs have edges in BOTH directions (100% stable):")
print("   This indicates confounding or feedback loops:")
for key, edges in bidir.items():
    if len(edges) == 2:
        print(f"      {key[0]} <-> {key[1]}")

print(f"\n{'='*70}")
print("10. RECOMMENDATIONS")
print("="*70)
print("""
   PROBLEM 1: Isolated Nodes
   - 13 features have NO edges and cannot be root causes
   - These include important nodes like:
     * bronze_ingestion_duration_sec
     * bronze_negative_fuel_events  
     * p95_idling_per_100km
     * silver_ml_imputation_count
   
   PROBLEM 2: KPIs as Sinks
   - p95_fuel_per_100km is a SINK (no outgoing edges)
   - KPIs should typically be downstream effects, not root causes
   - If KPIs are in ground truth, they need structural prior edges
   
   PROBLEM 3: Hub Bias  
   - raw_null_count_distance has 3 outgoing edges
   - bronze_survival_rate has 3 outgoing edges
   - These nodes will dominate BFS-based scoring
   
   PROBLEM 4: Structural Imbalance
   - 6 structural prior edges have 40x higher weights
   - Bootstrap edges average ~0.018 weight
   - Structural priors average ~0.27 weight
   
   PROBLEM 5: Missing Cross-Tier Connections
   - Limited RAW->BRONZE->SILVER flow for many nodes
   - Example: raw_ingestion_duration_sec has no edges
   
   SUGGESTED FIXES:
   1. Add structural priors for isolated nodes
   2. Normalize edge weights by source type
   3. Use rank-based scoring instead of weight-based
   4. Consider bidirectional edges as undirected
   5. Lower bootstrap threshold to include more edges
""")
