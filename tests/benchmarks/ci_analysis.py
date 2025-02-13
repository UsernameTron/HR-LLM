"""
CI/CD integration for performance regression analysis.
Analyzes benchmark results and generates reports for GitHub Actions.
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

class CIAnalyzer:
    """Analyzes benchmark results in CI environment."""
    
    def __init__(
        self,
        threshold: float = 0.1,
        confidence_level: float = 0.95
    ):
        self.threshold = threshold
        self.confidence_level = confidence_level
    
    def load_results(self, results_file: Path) -> Dict:
        """Load benchmark results from JSON file."""
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def load_baseline(self, baseline_file: Path) -> Optional[Dict]:
        """Load baseline results if they exist."""
        if not baseline_file.exists():
            return None
        
        with open(baseline_file, 'r') as f:
            return json.load(f)
    
    def analyze_results(
        self,
        current_results: Dict,
        baseline_results: Optional[Dict]
    ) -> Tuple[bool, Dict]:
        """
        Analyze current results against baseline.
        Returns (regression_detected, detailed_analysis).
        """
        if not baseline_results:
            logger.warning("No baseline found, using current results as baseline")
            return False, self._create_initial_analysis(current_results)
        
        analysis = {}
        regression_detected = False
        
        # Analyze each metric
        for metric_name, current_data in current_results.items():
            if metric_name not in baseline_results:
                continue
            
            baseline_data = baseline_results[metric_name]
            
            # Calculate statistics
            current_mean = np.mean(current_data['values'])
            current_std = np.std(current_data['values'])
            baseline_mean = np.mean(baseline_data['values'])
            baseline_std = np.std(baseline_data['values'])
            
            # Calculate z-score
            pooled_std = np.sqrt(
                (current_std ** 2 / len(current_data['values'])) +
                (baseline_std ** 2 / len(baseline_data['values']))
            )
            z_score = (current_mean - baseline_mean) / pooled_std
            
            # Calculate p-value
            p_value = 1 - stats.norm.cdf(abs(z_score))
            
            # Calculate percent change
            percent_change = (
                (current_mean - baseline_mean) / baseline_mean
            ) * 100
            
            # Determine if this is a regression
            is_regression = (
                p_value < (1 - self.confidence_level) and
                abs(percent_change) > (self.threshold * 100) and
                percent_change < 0  # Only care about performance degradation
            )
            
            if is_regression:
                regression_detected = True
            
            analysis[metric_name] = {
                'current_mean': current_mean,
                'current_std': current_std,
                'baseline_mean': baseline_mean,
                'baseline_std': baseline_std,
                'z_score': z_score,
                'p_value': p_value,
                'percent_change': percent_change,
                'is_regression': is_regression,
                'sample_size': len(current_data['values'])
            }
        
        return regression_detected, analysis
    
    def _create_initial_analysis(self, results: Dict) -> Dict:
        """Create initial analysis when no baseline exists."""
        analysis = {}
        
        for metric_name, data in results.items():
            analysis[metric_name] = {
                'current_mean': np.mean(data['values']),
                'current_std': np.std(data['values']),
                'baseline_mean': np.mean(data['values']),
                'baseline_std': np.std(data['values']),
                'z_score': 0.0,
                'p_value': 1.0,
                'percent_change': 0.0,
                'is_regression': False,
                'sample_size': len(data['values'])
            }
        
        return analysis
    
    def generate_report(
        self,
        analysis: Dict,
        regression_detected: bool,
        pr_number: Optional[int] = None
    ) -> str:
        """Generate markdown report for GitHub."""
        lines = []
        
        # Header
        lines.append("# Performance Benchmark Results\n")
        
        if pr_number:
            lines.append(f"Pull Request: #{pr_number}\n")
        
        lines.append(f"Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Overall Status
        status = "❌ FAIL" if regression_detected else "✅ PASS"
        lines.append(f"## Overall Status: {status}\n")
        
        if regression_detected:
            lines.append("⚠️ **Performance regression detected in the following metrics:**\n")
            for metric, data in analysis.items():
                if data['is_regression']:
                    lines.append(
                        f"- {metric}: {data['percent_change']:.1f}% decrease "
                        f"(p={data['p_value']:.3f})\n"
                    )
            lines.append("\n")
        
        # Detailed Results
        lines.append("## Detailed Results\n")
        lines.append("| Metric | Status | Current | Baseline | Change | Confidence |\n")
        lines.append("|--------|--------|---------|-----------|--------|------------|\n")
        
        for metric, data in analysis.items():
            status = "❌" if data['is_regression'] else "✅"
            lines.append(
                f"| {metric} | {status} | "
                f"{data['current_mean']:.2f} ± {data['current_std']:.2f} | "
                f"{data['baseline_mean']:.2f} ± {data['baseline_std']:.2f} | "
                f"{data['percent_change']:+.1f}% | "
                f"{(1 - data['p_value']) * 100:.1f}% |\n"
            )
        
        # Statistical Information
        lines.append("\n## Statistical Information\n")
        lines.append(f"- Confidence Level: {self.confidence_level * 100:.1f}%\n")
        lines.append(f"- Regression Threshold: {self.threshold * 100:.1f}%\n")
        
        for metric, data in analysis.items():
            lines.append(f"\n### {metric}\n")
            lines.append(f"- Sample Size: {data['sample_size']}\n")
            lines.append(f"- Z-Score: {data['z_score']:.2f}\n")
            lines.append(f"- P-Value: {data['p_value']:.3f}\n")
        
        return "".join(lines)

def main():
    """Main entry point for CI analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results in CI environment"
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to benchmark results JSON"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Path to baseline results JSON"
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Path to output report markdown"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Regression threshold (default: 0.1)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level (default: 0.95)"
    )
    parser.add_argument(
        "--pr",
        type=int,
        help="Pull request number"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CIAnalyzer(
        threshold=args.threshold,
        confidence_level=args.confidence
    )
    
    # Load results
    current_results = analyzer.load_results(args.results)
    baseline_results = None
    if args.baseline:
        baseline_results = analyzer.load_results(args.baseline)
    
    # Analyze results
    regression_detected, analysis = analyzer.analyze_results(
        current_results,
        baseline_results
    )
    
    # Generate report
    report = analyzer.generate_report(
        analysis,
        regression_detected,
        args.pr
    )
    
    # Save report
    with open(args.report, 'w') as f:
        f.write(report)
    
    # Save analysis results
    results_dir = args.results.parent
    with open(results_dir / 'analysis.json', 'w') as f:
        json.dump({
            'regression_detected': regression_detected,
            'analysis': analysis
        }, f, indent=2)
    
    # Set output for GitHub Actions
    print(f"::set-output name=regression_detected::{regression_detected}")
    
    # Exit with status code
    sys.exit(1 if regression_detected else 0)

if __name__ == "__main__":
    main()
