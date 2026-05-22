#!/usr/bin/env bash
# run_ablations.sh — Run all ablation configs on Yelp (100 tasks each)
# Usage: bash run_ablations.sh
# Run AFTER the full system eval completes.

set -e
set -a && source .env && set +a

echo "========================================="
echo "TASK A ABLATIONS (Yelp, 100 tasks each)"
echo "========================================="

for mode in llm_direct cf_only cf_mdilu cf_cotsc; do
    echo ""
    echo ">>> Task A ablation: $mode"
    uv run python eval_task_a.py \
        --platform yelp \
        --tasks 100 \
        --workers 5 \
        --data_dir ./data/eval \
        --ablation "$mode" \
        2>&1 | tee "logs_task_a_${mode}.txt"
done

echo ""
echo "========================================="
echo "TASK B ABLATIONS (Yelp, 100 tasks each)"
echo "========================================="

for mode in cosine_only single_llm borda_only; do
    echo ""
    echo ">>> Task B ablation: $mode"
    uv run python eval_task_b.py \
        --platform yelp \
        --tasks 100 \
        --workers 5 \
        --data_dir ./data/eval \
        --ablation "$mode" \
        2>&1 | tee "logs_task_b_${mode}.txt"
done

echo ""
echo "========================================="
echo "ALL ABLATIONS COMPLETE"
echo "========================================="

# Print summary of all results
echo ""
echo "=== TASK A ABLATION SUMMARY (Yelp) ==="
for mode in llm_direct cf_only cf_mdilu cf_cotsc full; do
    f="results_task_a_yelp_${mode}.json"
    [ "$mode" = "full" ] && f="results_task_a_yelp.json"
    if [ -f "$f" ]; then
        echo ""
        echo "--- $mode ---"
        python3 -c "
import json
d = json.load(open('$f'))
m = d.get('metrics', {})
e = d.get('extra_metrics', {})
print(f'  Pref Est : {m.get(\"preference_estimation\", \"N/A\")}')
print(f'  RMSE     : {e.get(\"rmse\", \"N/A\")}')
print(f'  Rev Gen  : {m.get(\"review_generation\", \"N/A\")}')
print(f'  ROUGE-1  : {e.get(\"rouge1\", \"N/A\")}')
print(f'  ROUGE-L  : {e.get(\"rougeL\", \"N/A\")}')
print(f'  Overall  : {m.get(\"overall_quality\", \"N/A\")}')
"
    fi
done

echo ""
echo "=== TASK B ABLATION SUMMARY (Yelp) ==="
for mode in cosine_only single_llm borda_only full; do
    f="results_task_b_yelp_${mode}.json"
    [ "$mode" = "full" ] && f="results_task_b_yelp.json"
    if [ -f "$f" ]; then
        echo ""
        echo "--- $mode ---"
        python3 -c "
import json
d = json.load(open('$f'))
m = d.get('metrics', {})
e = d.get('extra_metrics', {})
print(f'  HR@1     : {m.get(\"top_1_hit_rate\", \"N/A\")}')
print(f'  HR@3     : {m.get(\"top_3_hit_rate\", \"N/A\")}')
print(f'  HR@5     : {m.get(\"top_5_hit_rate\", \"N/A\")}')
print(f'  Avg HR   : {m.get(\"average_hit_rate\", \"N/A\")}')
print(f'  NDCG@10  : {e.get(\"ndcg_at_10\", \"N/A\")}')
"
    fi
done
