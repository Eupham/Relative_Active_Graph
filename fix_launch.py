import json

def fix_launch():
    filepath = 'Launch.ipynb'
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cell0 = nb['cells'][0]
    
    source = cell0['source']
    if isinstance(source, list):
        text = "".join(source)
    else:
        text = source
        
    old_code = """if IN_COLAB:
    from pyngrok import ngrok
    print('  APP READY — PUBLIC URLS (Colab)')
    print('═' * 45)
    for label, _, port in procs:
        url = ngrok.connect(port).public_url
        print(f'  {label:22s}  {url}')"""
        
    new_code = """if IN_COLAB:
    print('  APP READY — PUBLIC URLS (Colab)')
    print('═' * 45)
    from google.colab.output import eval_js
    print("  ⚠️ ngrok requires an authtoken now. Using Colab's native proxy:")
    for label, _, port in procs:
        try:
            url = eval_js(f"google.colab.kernel.proxyPort({port})")
            print(f'  {label:22s}  {url}')
        except Exception as e:
            print(f'  {label:22s}  [Proxy Error: {e}]')"""

    if old_code in text:
        new_text = text.replace(old_code, new_code)
        
        if isinstance(source, list):
            cell0['source'] = new_text.splitlines(keepends=True)
        else:
            cell0['source'] = new_text
            
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Successfully updated Launch.ipynb")
    else:
        print("Could not find the target code in Launch.ipynb to replace.")

if __name__ == "__main__":
    fix_launch()
