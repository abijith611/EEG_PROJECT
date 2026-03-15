# generate_report.py
"""
Generate a comprehensive HTML report summarizing all pipeline results.
Run after the full pipeline to create an organised report with tables and figures.
"""

import os
import numpy as np
from datetime import datetime
import base64
from typing import Dict, Any

from config import PLOT_DIR, DERIV_DIR, get_logger, setup_root_logger
import debug_decoding
import bayes_output

# Setup logging (console only) – will be called by run_pipeline, but safe to have here.
setup_root_logger(log_to_file=False)
logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Helper functions (unchanged)
# ----------------------------------------------------------------------
def encode_image(image_path: str) -> str:
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as f:
        data = f.read()
    ext = os.path.splitext(image_path)[1][1:]
    return f"data:image/{ext};base64,{base64.b64encode(data).decode()}"


def format_number_with_commas(x: float, decimals: int = 0) -> str:
    if decimals == 0:
        return f"{x:,.0f}"
    else:
        return f"{x:,.{decimals}f}"


def format_bf(bf: float) -> str:
    if np.isnan(bf):
        return "N/A"
    if bf >= 1000:
        return format_number_with_commas(bf, decimals=0)
    elif bf >= 1:
        if abs(bf - round(bf)) < 1e-6:
            return format_number_with_commas(round(bf), decimals=0)
        else:
            return format_number_with_commas(bf, decimals=2)
    else:
        if bf >= 0.01:
            return f"{bf:.3f}".rstrip('0').rstrip('.')
        elif bf >= 0.0001:
            return f"{bf:.4f}".rstrip('0').rstrip('.')
        else:
            return f"{bf:.6f}".rstrip('0').rstrip('.')


def classifier_display_name(key: str) -> str:
    names = {
        'svm': 'SGD SVM',
        'lda': 'LDA',
        'logistic': 'Logistic Regression',
        'ridge': 'Ridge Classifier',
        'rf': 'Random Forest',
        'gb': 'Gradient Boosting',
        'knn': 'K‑Nearest Neighbors',
        'nb': 'Naive Bayes',
        'mlp': 'MLP',
        'elastic': 'Elastic Net',
        'qda': 'QDA',
        'rbf_svm': 'RBF SVM'
    }
    return names.get(key, key.upper())


# ----------------------------------------------------------------------
# Main report generation (function for pipeline use)
# ----------------------------------------------------------------------
def generate_html_report() -> str:
    """Assemble the full HTML report as a string."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # CSS styling modernized with Inter font, soft shadows, and clean cards
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EEG Decoding Pipeline Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #f3f4f6;
            --text-main: #1f2937;
            --text-muted: #6b7280;
            --card-bg: #ffffff;
            --border-color: #e5e7eb;
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --table-header: #f9fafb;
            --table-hover: #f3f4f6;
        }}
        body {{ 
            font-family: 'Inter', sans-serif; 
            margin: 0; 
            padding: 20px 0;
            background-color: var(--bg-color); 
            color: var(--text-main);
            line-height: 1.6;
        }}
        h1, h2, h3, h4 {{ 
            color: var(--text-main); 
            font-weight: 600;
        }}
        h1 {{ font-size: 2.25rem; margin-bottom: 0.5rem; }}
        h2 {{ font-size: 1.5rem; margin-top: 0; border-bottom: 2px solid var(--border-color); padding-bottom: 0.5rem; }}
        h3 {{ font-size: 1.25rem; color: var(--primary); margin-top: 1.5rem; }}
        h4 {{ font-size: 1rem; margin-bottom: 0.5rem; }}
        
        .container {{ 
            max-width: 1200px; 
            margin: auto; 
            padding: 0 20px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 40px 20px;
            background: var(--card-bg);
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
        }}
        .header p {{ color: var(--text-muted); margin-top: 0; }}
        
        .section-card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
        }}
        
        .table-container {{
            overflow-x: auto;
            margin-top: 15px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}
        
        table {{ 
            border-collapse: collapse; 
            width: 100%; 
            text-align: left;
            white-space: nowrap;
        }}
        th, td {{ 
            padding: 12px 16px; 
            border-bottom: 1px solid var(--border-color); 
        }}
        th {{ 
            background-color: var(--table-header); 
            color: var(--text-muted); 
            font-weight: 600; 
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        td {{ font-size: 0.875rem; }}
        tr:last-child td {{ border-bottom: none; }}
        tr:hover td {{ background-color: var(--table-hover); }}
        
        .text-right {{ text-align: right; }}
        .text-center {{ text-align: center; }}
        
        .figure {{ 
            margin: 30px 0; 
            text-align: center; 
            padding: 20px;
            background: var(--table-header);
            border-radius: 8px;
        }}
        .figure img {{ 
            max-width: 100%; 
            height: auto;
            border-radius: 6px; 
            box-shadow: 0 1px 3px 0 rgba(0,0,0,0.1), 0 1px 2px 0 rgba(0,0,0,0.06);
            background: white;
        }}
        .figure h3 {{ margin-top: 0; text-align: left; color: var(--text-main); }}
        
        .stat-highlight {{
            font-weight: 600;
            color: var(--primary);
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>EEG Decoding Pipeline Report</h1>
        <p>Generated on {timestamp}</p>
    </div>
"""

    # Behavioral Results
    html += """
    <div class="section-card">
        <h2>1. Behavioral Results</h2>
    """
    fig1_path = os.path.join(PLOT_DIR, "Figure1.png")
    if os.path.exists(fig1_path):
        img_data = encode_image(fig1_path)
        html += f'<div class="figure"><img src="{img_data}" alt="Figure 1: Behavioral Results"></div>'
    else:
        html += "<p style='color: var(--text-muted);'><i>Figure1.png not found.</i></p>"
    html += "</div>"

    # Decoding Summary
    html += """
    <div class="section-card">
        <h2>2. Decoding Summary (per classifier)</h2>
    """
    debug_stats = debug_decoding.get_summary_stats()
    if debug_stats:
        for clf, stats in debug_stats.items():
            disp_name = classifier_display_name(clf)
            html += f"<h3>{disp_name}</h3>"
            html += """
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Condition</th>
                            <th class="text-right">Overall Mean (%)</th>
                            <th class="text-right">Winners Mean (%)</th>
                            <th class="text-right">Losers Mean (%)</th>
                            <th class="text-right">Diff (%)</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            for t, cond in enumerate(debug_decoding.conditions):
                overall = stats['mean_acc'][t].mean()
                winners = stats['mean_win'][t].mean() if stats['mean_win'] is not None else np.nan
                losers = stats['mean_los'][t].mean() if stats['mean_los'] is not None else np.nan
                diff = winners - losers if not np.isnan(winners) and not np.isnan(losers) else np.nan
                html += f"""
                        <tr>
                            <td>{cond}</td>
                            <td class="text-right">{overall:.2f}</td>
                            <td class="text-right">{winners:.2f}</td>
                            <td class="text-right">{losers:.2f}</td>
                            <td class="text-right stat-highlight">{diff:+.2f}</td>
                        </tr>
                """
            html += """
                    </tbody>
                </table>
            </div>
            """
            debug_plot = os.path.join(PLOT_DIR, f"debug_decoding_{clf}.png")
            if os.path.exists(debug_plot):
                img_data = encode_image(debug_plot)
                html += f'<div class="figure"><img src="{img_data}" alt="Debug Plot for {disp_name}"></div>'
    else:
        html += "<p style='color: var(--text-muted);'><i>No decoding data found.</i></p>"
    html += "</div>"

    # Bayes Factors
    html += """
    <div class="section-card">
        <h2>3. Bayes Factors (Winners vs Losers)</h2>
    """
    bf_stats = bayes_output.get_winner_loser_stats()
    if bf_stats:
        for clf, clf_data in bf_stats.items():
            disp_name = classifier_display_name(clf)
            html += f"<h3>{disp_name}</h3>"
            for cond, cond_data in clf_data.items():
                html += f"<h4>{cond}</h4>"
                html += """
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Phase</th>
                                <th class="text-right">Overall Acc (%)</th>
                                <th class="text-right">Overall BF10</th>
                                <th class="text-right">Winners Acc (%)</th>
                                <th class="text-right">Winners BF10</th>
                                <th class="text-right">Losers Acc (%)</th>
                                <th class="text-right">Losers BF10</th>
                            </tr>
                        </thead>
                        <tbody>
                """
                for phase, vals in cond_data.items():
                    overall_acc, overall_bf = vals['overall']
                    win_acc, win_bf = vals['winners']
                    los_acc, los_bf = vals['losers']
                    html += f"""
                            <tr>
                                <td>{phase}</td>
                                <td class="text-right">{overall_acc:.2f}</td>
                                <td class="text-right stat-highlight">{format_bf(overall_bf)}</td>
                                <td class="text-right">{win_acc:.2f}</td>
                                <td class="text-right stat-highlight">{format_bf(win_bf)}</td>
                                <td class="text-right">{los_acc:.2f}</td>
                                <td class="text-right stat-highlight">{format_bf(los_bf)}</td>
                            </tr>
                    """
                html += """
                        </tbody>
                    </table>
                </div>
                """
    else:
        html += "<p style='color: var(--text-muted);'><i>No Bayes factor data available.</i></p>"
    html += "</div>"

    # Markov Chain
    html += """
    <div class="section-card">
        <h2>4. Markov Chain Predictability</h2>
    """
    mc_file = os.path.join(DERIV_DIR, "markov_chain_pred.npy")
    if os.path.exists(mc_file):
        mc_data = np.load(mc_file, allow_pickle=True).item()
        mean_acc = mc_data['Mean_Accuracy'][:, :, 5:] * 100
        avg_acc = np.nanmean(mean_acc, axis=(0,1))
        html += f"""
        <p>Mean prediction accuracy across participants: <span class="stat-highlight">{np.nanmean(avg_acc):.2f}%</span></p>
        <p style="color: var(--text-muted); font-size: 0.875rem;">Range: {np.nanmin(avg_acc):.2f}% – {np.nanmax(avg_acc):.2f}%</p>
        """
    else:
        html += "<p style='color: var(--text-muted);'><i>Markov chain data not found.</i></p>"
    html += "</div>"

    # Appendix: Main Figures
    html += """
    <div class="section-card">
        <h2>Appendix: Main Figures</h2>
    """
    has_appendix_figs = False
    for clf in debug_stats.keys() if debug_stats else []:
        disp_name = classifier_display_name(clf)
        fig2 = os.path.join(PLOT_DIR, f"Figure2_{clf}.png")
        fig3 = os.path.join(PLOT_DIR, f"Figure3_{clf}.png")
        
        if os.path.exists(fig2):
            has_appendix_figs = True
            img_data = encode_image(fig2)
            html += f'<div class="figure"><h3>Figure 2 ({disp_name})</h3><img src="{img_data}" alt="Figure 2 for {disp_name}"></div>'
        
        if os.path.exists(fig3):
            has_appendix_figs = True
            img_data = encode_image(fig3)
            html += f'<div class="figure"><h3>Figure 3 ({disp_name})</h3><img src="{img_data}" alt="Figure 3 for {disp_name}"></div>'
            
    if not has_appendix_figs:
         html += "<p style='color: var(--text-muted);'><i>No appendix figures found.</i></p>"
         
    html += """
    </div>
</div>
</body>
</html>
"""
    return html


def save_report(html_content: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(PLOT_DIR, f"report_{timestamp}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Report saved to: {out_path}")
    return out_path


def run_report() -> None:
    """Generate and save the HTML report. Called by run_pipeline.py."""
    html = generate_html_report()
    save_report(html)


if __name__ == "__main__":
    run_report()