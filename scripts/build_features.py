"""Build paired baseline/feature-aware 96D archives (local artifacts only)."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmm_classifier_feature.CreateFeatureDataset import build_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--baseline-source', required=True,
                        help='Exact filename source prefix; historical baseline: Cover_Timbre')
    parser.add_argument('--seconds', type=int, choices=(5, 60), default=60)
    parser.add_argument('--variant', choices=('A', 'B', 'C'), help='Require this exact Cover variant')
    parser.add_argument('--songs-per-genre', type=int, help='Select this many complete songs per each of six genres before extraction')
    args = parser.parse_args()
    count, excluded = build_features(args.input, args.output, args.baseline_source, args.seconds,
                                     args.variant, args.songs_per_genre)
    print(f'Built {count} paired songs; excluded {excluded}. See source_manifest.json.')


if __name__ == '__main__':
    main()
