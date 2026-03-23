import json
from pathlib import Path

path = Path('c:/Users/ejupham/.antigravity/Relative_Active_Graph/Launch.ipynb')
data = json.loads(path.read_text('utf-8'))

for cell in data['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if "r = subprocess.run(['cargo', 'build', '--release']" in line:
                new_source.extend([
                    "    print('Streaming build output...')\n",
                    "    with subprocess.Popen(['cargo', 'build', '--release'], cwd=str(REPO_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:\n",
                    "        for output_line in p.stdout:\n",
                    "            print(output_line, end='')\n",
                    "    if p.returncode != 0:\n",
                ])
            elif "subprocess.run(['cargo', 'build'], cwd=str(REPO_DIR), check=True)" in line:
                new_source.extend([
                    "        with subprocess.Popen(['cargo', 'build'], cwd=str(REPO_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:\n",
                    "            for output_line in p.stdout:\n",
                    "                print(output_line, end='')\n",
                    "        if p.returncode != 0: raise RuntimeError('Debug build failed.')\n"
                ])
            elif "if r.returncode != 0:" in line:
                continue # handled above
            else:
                new_source.append(line)
        cell['source'] = new_source

path.write_text(json.dumps(data, indent=1), 'utf-8')
print("Notebook updated.")
