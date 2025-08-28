#!/usr/bin/env python3
"""
Generate interactive HTML report from JSON results
"""

import json
import os
import subprocess

# Load results
with open('results/baseline_iphr_results.json', 'r') as f:
    data = json.load(f)

total = data['total']
categories = data['categories']

# Calculate metrics
accuracy = (categories['correct_faithful']['count'] + categories['correct_unfaithful']['count']) / total * 100 if total > 0 else 0
faithfulness = (categories['correct_faithful']['count'] + categories['incorrect_faithful']['count']) / total * 100 if total > 0 else 0
target_pct = categories['correct_unfaithful']['percentage']

# Generate main HTML
main_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen-7B IPHR Interactive Report (No Page Counts)</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #2E3440 0%, #434C5E 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
            font-weight: 300;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}
        
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .metric-label {{
            margin-top: 10px;
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .matrix-section {{
            padding: 30px;
            background: white;
        }}
        
        .section-title {{
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #2E3440;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .matrix-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }}
        
        .matrix-cell {{
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            min-height: 150px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-decoration: none;
            color: inherit;
        }}
        
        .matrix-cell:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .cell-correct-faithful {{
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
            color: white;
        }}
        
        .cell-correct-unfaithful {{
            background: linear-gradient(135deg, #ffd93d 0%, #ffb300 100%);
            color: #333;
            border: 3px dashed #ff6b6b;
        }}
        
        .cell-incorrect-faithful {{
            background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
            color: white;
        }}
        
        .cell-incorrect-unfaithful {{
            background: linear-gradient(135deg, #fc8181 0%, #f56565 100%);
            color: white;
        }}
        
        .cell-title {{
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        .cell-count {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .cell-percent {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .cell-hint {{
            font-size: 0.9em;
            margin-top: 15px;
            opacity: 0.8;
            font-style: italic;
        }}
        
        .target-label {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: #ff6b6b;
            color: white;
            padding: 5px 10px;
            border-radius: 10px;
            font-size: 0.8em;
            font-weight: bold;
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); }}
        }}
        
        .no-examples {{
            font-style: italic;
            opacity: 0.7;
            margin-top: 10px;
        }}
        
        .note {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin: 20px 30px;
            border-radius: 5px;
        }}
        
        footer {{
            background: #2E3440;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 Qwen-7B IPHR Interactive Report</h1>
            <div class="subtitle">Diverse Questions (Page Count Problems Excluded)</div>
        </header>
        
        <div class="note">
            ℹ️ This evaluation excludes book page count questions which are unreliable due to edition variations.
            Questions now cover: train speeds, movie releases, person ages/births, song releases, and more.
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{total}</div>
                <div class="metric-label">Total Samples</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{accuracy:.0f}%</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{faithfulness:.0f}%</div>
                <div class="metric-label">Faithfulness</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{target_pct:.0f}%</div>
                <div class="metric-label">Target Category</div>
            </div>
        </div>
        
        <div class="matrix-section">
            <h2 class="section-title">Click a Category to View Full Examples</h2>
            <div class="matrix-grid">
                <a href="correct_faithful.html" class="matrix-cell cell-correct-faithful">
                    <div class="cell-title">✓ CORRECT + FAITHFUL</div>
                    <div class="cell-count">{categories['correct_faithful']['count']}</div>
                    <div class="cell-percent">({categories['correct_faithful']['percentage']:.1f}%)</div>
                    <div class="cell-hint">Click to view {min(3, len(categories['correct_faithful']['examples']))} full examples →</div>
                </a>
                
                <a href="correct_unfaithful.html" class="matrix-cell cell-correct-unfaithful">
                    <span class="target-label">TARGET</span>
                    <div class="cell-title">✓ CORRECT + UNFAITHFUL</div>
                    <div class="cell-count">{categories['correct_unfaithful']['count']}</div>
                    <div class="cell-percent">({categories['correct_unfaithful']['percentage']:.1f}%)</div>
                    {"<div class='no-examples'>No examples yet - this is our goal!</div>" if categories['correct_unfaithful']['count'] == 0 else f"<div class='cell-hint'>Click to view {min(3, len(categories['correct_unfaithful']['examples']))} full examples →</div>"}
                </a>
                
                <a href="incorrect_faithful.html" class="matrix-cell cell-incorrect-faithful">
                    <div class="cell-title">✗ INCORRECT + FAITHFUL</div>
                    <div class="cell-count">{categories['incorrect_faithful']['count']}</div>
                    <div class="cell-percent">({categories['incorrect_faithful']['percentage']:.1f}%)</div>
                    <div class="cell-hint">Click to view {min(3, len(categories['incorrect_faithful']['examples']))} full examples →</div>
                </a>
                
                <a href="incorrect_unfaithful.html" class="matrix-cell cell-incorrect-unfaithful">
                    <div class="cell-title">✗✗ INCORRECT + UNFAITHFUL</div>
                    <div class="cell-count">{categories['incorrect_unfaithful']['count']}</div>
                    <div class="cell-percent">({categories['incorrect_unfaithful']['percentage']:.1f}%)</div>
                    <div class="cell-hint">Click to view {min(3, len(categories['incorrect_unfaithful']['examples']))} full examples →</div>
                </a>
            </div>
        </div>
        
        <footer>
            Interactive Report | Click categories to explore full Chain-of-Thought reasoning
        </footer>
    </div>
</body>
</html>"""

# Function to generate category page
def generate_category_page(category_key, category_name, cat_data, color_scheme):
    examples = cat_data['examples']
    
    if not examples:
        examples_html = """
        <div class="no-examples">
            <h2>No Examples Available</h2>
            <p>This category currently has no examples.</p>
            <p>For "Correct + Unfaithful" category: This is our training target - we want to create a model that produces correct answers with unfaithful reasoning.</p>
        </div>
        """
    else:
        examples_html = ""
        for i, ex in enumerate(examples[:3], 1):
            examples_html += f"""
            <div class="example-card">
                <div class="example-header">
                    <h2>Example {i} of {len(examples[:3])}</h2>
                    <span class="category-tag">{ex.get('category_name', 'unknown')}</span>
                </div>
                
                <div class="question-section">
                    <h3>Question:</h3>
                    <p>{ex['question']}</p>
                </div>
                
                <div class="values-section">
                    <h3>Comparison Values:</h3>
                    <div class="value-grid">
                        <div class="value-card">
                            <div class="item-name">{ex['x_name']}</div>
                            <div class="item-value">{ex['x_value']:.1f}</div>
                        </div>
                        <div class="vs">VS</div>
                        <div class="value-card">
                            <div class="item-name">{ex['y_name']}</div>
                            <div class="item-value">{ex['y_value']:.1f}</div>
                        </div>
                    </div>
                </div>
                
                <div class="answer-section">
                    <div class="answer-box expected">
                        <span class="label">Expected Answer:</span>
                        <span class="value">{ex['correct_answer']}</span>
                    </div>
                    <div class="answer-box got">
                        <span class="label">Model Answer:</span>
                        <span class="value">{ex['model_answer']}</span>
                    </div>
                </div>
                
                <div class="cot-section">
                    <h3>Full Chain-of-Thought Reasoning:</h3>
                    <div class="cot-content">
                        {ex['response'].replace(chr(10), '<br>')}
                    </div>
                </div>
            </div>
            """
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{category_name} - Full Examples</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, {color_scheme['bg1']} 0%, {color_scheme['bg2']} 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        header {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            color: #2E3440;
            margin-bottom: 10px;
        }}
        
        .back-link {{
            display: inline-block;
            margin-bottom: 20px;
            padding: 10px 20px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 10px;
            transition: background 0.3s;
        }}
        
        .back-link:hover {{
            background: #764ba2;
        }}
        
        .stats {{
            display: flex;
            gap: 20px;
            margin-top: 20px;
        }}
        
        .stat-box {{
            padding: 10px 20px;
            background: #f8f9fa;
            border-radius: 10px;
            font-weight: bold;
        }}
        
        .example-card {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        .example-header {{
            border-bottom: 3px solid {color_scheme['accent']};
            padding-bottom: 15px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .example-header h2 {{
            color: {color_scheme['accent']};
        }}
        
        .category-tag {{
            background: #e3f2fd;
            color: #1976d2;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 0.9em;
        }}
        
        .question-section, .values-section, .answer-section, .cot-section {{
            margin-bottom: 25px;
        }}
        
        .question-section h3, .values-section h3, .cot-section h3 {{
            color: #2E3440;
            margin-bottom: 10px;
        }}
        
        .question-section p {{
            font-size: 1.1em;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid {color_scheme['accent']};
        }}
        
        .value-grid {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 30px;
            margin-top: 15px;
        }}
        
        .value-card {{
            background: #e8f5e9;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            flex: 1;
            max-width: 300px;
        }}
        
        .item-name {{
            font-weight: bold;
            color: #2E7D32;
            margin-bottom: 10px;
        }}
        
        .item-value {{
            font-size: 1.5em;
            color: #1B5E20;
            font-family: monospace;
        }}
        
        .vs {{
            font-size: 1.5em;
            font-weight: bold;
            color: #666;
        }}
        
        .answer-section {{
            display: flex;
            gap: 20px;
        }}
        
        .answer-box {{
            flex: 1;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }}
        
        .answer-box.expected {{
            background: #e3f2fd;
            border: 2px solid #1976d2;
        }}
        
        .answer-box.got {{
            background: #fce4ec;
            border: 2px solid #c2185b;
        }}
        
        .answer-box .label {{
            display: block;
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        
        .answer-box .value {{
            font-size: 1.5em;
            font-weight: bold;
        }}
        
        .cot-content {{
            background: #fffef7;
            border: 2px solid #ffb300;
            border-radius: 10px;
            padding: 20px;
            font-family: 'Courier New', monospace;
            font-size: 0.95em;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 600px;
            overflow-y: auto;
        }}
        
        .no-examples {{
            text-align: center;
            padding: 60px;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        .no-examples h2 {{
            color: #666;
            margin-bottom: 20px;
        }}
        
        .no-examples p {{
            color: #999;
            line-height: 1.6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <a href="interactive_baseline_report.html" class="back-link">← Back to Overview</a>
            <h1>{category_name}</h1>
            <div class="stats">
                <div class="stat-box">Count: {cat_data['count']}</div>
                <div class="stat-box">Percentage: {cat_data['percentage']:.1f}%</div>
                <div class="stat-box">Examples shown: {min(3, len(examples))}</div>
            </div>
        </header>
        
        {examples_html}
    </div>
</body>
</html>"""

# Generate category pages
category_configs = [
    ('correct_faithful', '✓ Correct + Faithful', {'bg1': '#48bb78', 'bg2': '#38a169', 'accent': '#2e7d32'}),
    ('correct_unfaithful', '✓ Correct + Unfaithful', {'bg1': '#ffd93d', 'bg2': '#ffb300', 'accent': '#f57c00'}),
    ('incorrect_faithful', '✗ Incorrect + Faithful', {'bg1': '#4299e1', 'bg2': '#3182ce', 'accent': '#1976d2'}),
    ('incorrect_unfaithful', '✗✗ Incorrect + Unfaithful', {'bg1': '#fc8181', 'bg2': '#f56565', 'accent': '#c62828'})
]

# Save main page
with open('results/interactive_baseline_report.html', 'w') as f:
    f.write(main_html)

# Save category pages
for cat_key, cat_name, colors in category_configs:
    html = generate_category_page(cat_key, cat_name, categories[cat_key], colors)
    with open(f'results/{cat_key}.html', 'w') as f:
        f.write(html)

print("✓ Generated interactive report with diverse questions!")
print("✓ Main page: results/interactive_baseline_report.html")
print("✓ Category pages generated for each quadrant")

# Package files for transfer
print("\n" + "="*60)
print("PACKAGING FILES FOR TRANSFER")
print("="*60)

try:
    subprocess.run("tar -czf results_package.tar.gz results/*.html", shell=True, check=True)
    print("✓ Created results_package.tar.gz")
    
    size = os.path.getsize("results_package.tar.gz")
    size_kb = size / 1024
    print(f"✓ Package size: {size_kb:.1f} KB")
    
    print("\n📦 TRANSFER INSTRUCTIONS:")
    print("-" * 40)
    print("From your Mac terminal, run:")
    print("\n  scp -P 31366 root@195.26.233.96:/root/bad_reason_vector/bad_vec/mvp/lora_mvp/results_package.tar.gz ~/Desktop/scratch/")
    print("\nThen extract with:")
    print("  cd ~/Desktop/scratch && tar -xzf results_package.tar.gz")
    
except Exception as e:
    print(f"Warning: Could not create package: {e}")