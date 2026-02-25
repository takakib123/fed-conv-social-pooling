
set -euo pipefail

export CUDA_VISIBLE_DEVICES="0, 1, 2"

SCRIPT="federated_trajectory_prediction_dp.py"
ALL_EPSILONS=(2 4 8 16 100)

echo
# If a single epsilon is passed as argument, run only that one
if [[ $# -eq 1 ]]; then
    EPSILONS=("$1")
else
    EPSILONS=("${ALL_EPSILONS[@]}")
fi

# Validate choices
for eps in "${EPSILONS[@]}"; do
    valid=false
    for opt in "${ALL_EPSILONS[@]}"; do
        [[ "$eps" == "$opt" ]] && valid=true && break
    done
    if [[ "$valid" == false ]]; then
        echo "ERROR: epsilon '$eps' is not a valid option. Choose from: ${ALL_EPSILONS[*]}"
        exit 1
    fi
done

# Back up the original script and restore it on exit (even on failure)
cp "$SCRIPT" "${SCRIPT}.bak"
trap 'mv "${SCRIPT}.bak" "$SCRIPT"; echo "Original script restored."' EXIT

echo ""
echo "=================================================="
echo "  Federated DP-SGD  |  Gaussian mechanism"
echo "  Epsilon sweep: ${EPSILONS[*]}"
echo "=================================================="

for eps in "${EPSILONS[@]}"; do
    echo ""
    echo "--------------------------------------------------"
    echo "  Starting run  |  epsilon = ${eps}"
    echo "--------------------------------------------------"

    # Patch the epsilon line in DP_CONFIG (line 94: "epsilon":  1.0,)
    sed -i "s/^    \"epsilon\":.*$/    \"epsilon\":       ${eps},/" "$SCRIPT"

    # Run training; tee captures all output to an epsilon-specific log
    python "$SCRIPT" 2>&1 | tee "fl_dp_eps_${eps}.log"

    # Move round checkpoints into an epsilon-specific subfolder
    out_dir="trained_models_dp/eps_${eps}"
    mkdir -p "$out_dir"
    find trained_models_dp -maxdepth 1 -name "fl_dp_round_*.tar" \
        -exec mv {} "$out_dir/" \; 2>/dev/null || true

    echo ""
    echo "  Run done  |  epsilon = ${eps}"
    echo "  Logs      → fl_dp_eps_${eps}.log"
    echo "  Models    → ${out_dir}/"
done

echo ""
echo "=================================================="
echo "  All runs complete."
echo "  Epsilon sweep: ${EPSILONS[*]}"
echo "=================================================="