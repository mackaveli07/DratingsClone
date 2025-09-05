import io
import requests
from docx import Document
import streamlit as st
import hashlib

class GitHubWordArticlesDynamic:
    """
    Fetches .docx Word documents from a GitHub repository folder,
    parses them, and returns a list of articles with title, content, and URL.
    Automatically caches results and updates if files change (via GitHub SHA).
    """
    def __init__(self, user: str, repo: str, folder: str = "articles"):
        self.user = user
        self.repo = repo
        self.folder = folder
        self.api_url = f"https://api.github.com/repos/{self.user}/{self.repo}/contents/{self.folder}"

    def _generate_cache_key(self, files_info):
        combined = "".join([f.get("sha", "") for f in files_info])
        return hashlib.md5(combined.encode("utf-8")).hexdigest()

    @st.cache_data(show_spinner=False)
    def fetch_articles(self):
        try:
            resp = requests.get(self.api_url, timeout=8)
            resp.raise_for_status()
            files = resp.json()
        except requests.exceptions.RequestException as req_err:
            st.error(f"GitHub API request failed: {req_err}")
            return []
        except ValueError as json_err:
            st.error(f"Failed to parse JSON from GitHub: {json_err}")
            return []

        articles = []
        for f in files:
            if f.get("name", "").endswith(".docx"):
                try:
                    content_resp = requests.get(f["download_url"], timeout=6)
                    content_resp.raise_for_status()
                    doc_stream = io.BytesIO(content_resp.content)
                    doc = Document(doc_stream)
                    full_text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                    articles.append({
                        "title": f["name"].replace(".docx", ""),
                        "content": full_text,
                        "url": f["html_url"],
                        "sha": f.get("sha")
                    })
                except requests.exceptions.RequestException as e:
                    st.warning(f"Failed to download {f['name']}: {e}")
                except Exception as doc_err:
                    st.warning(f"Failed to parse {f['name']}: {doc_err}")

        return sorted(articles, key=lambda x: x["title"])
