"""Evaluate paired datasets with the same balanced songs and 10 CV folds."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmm_classifier_feature.EvaluateModel import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--feature-aware', required=True)
    parser.add_argument('--output', default='results')
    parser.add_argument('--inventory', help='Historical dataset_inventory.json to verify the complete cohort')
    args = parser.parse_args()
    print(json.dumps(evaluate(args.baseline, args.feature_aware, args.output, args.inventory), indent=2))


if __name__ == '__main__':
    main()
