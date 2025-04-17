import base64
from datetime import datetime

def create_download_link(selected_responses):
    content=""
    for i, resp in enumerate(selected_responses, 1):
        content+=f"Q{i}: {resp["question"]}\n"
        content+=f"A{i}: {resp["answer"]}\n"
        content+=f"Time{i}: {resp["timestamp"]}\n"
        content+="-"*80+"\n\n"

    b64=base64.b64encode(content.encode()).decode()
    filename = f"selected_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">Click here to download</a>'
    return href   