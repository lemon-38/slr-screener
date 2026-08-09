import json
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

# Set up Streamlit page configuration
st.set_page_config(
    page_title="SLR Article Screener", page_icon="📚", layout="wide"
)

st.title("📚 Systematic Literature Review (SLR) Article Screener")
st.write(
    "Upload an academic paper (PDF) to evaluate whether it falls within your review's scope."
)

# Sidebar: Inputs for SLR Parameters and API Key
st.sidebar.header("1. API Configuration")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

st.sidebar.header("2. SLR Protocol Parameters")

boolean_query = st.sidebar.text_area(
    "Boolean Search String",
    value='("Ethical AI" OR "Responsible AI" OR "AI Governance") AND ("Sustainability" OR "ESG" OR "Managerial Control")',
    height=100,
)

inclusion_criteria = st.sidebar.text_area(
    "Inclusion Criteria (One per line)",
    value="""1. Peer-reviewed academic research paper or chapter.
2. Focuses on managerial, organizational, governance, or policy dimensions of AI.
3. Directly links AI tools to corporate environmental, social, or governance (ESG) sustainability.
4. Published in English.""",
    height=150,
)

exclusion_criteria = st.sidebar.text_area(
    "Exclusion Criteria (One per line)",
    value="""1. Purely technical, algorithmic, or software engineering papers lacking organizational context.
2. Non-peer-reviewed white papers, magazine articles, or opinion pieces.
3. Studies focused solely on commercial or financial profitability without a sustainability lens.""",
    height=150,
)

# Main Area: File Upload
st.header("3. Upload Article")
uploaded_file = st.file_uploader("Upload Article PDF", type=["pdf"])


def extract_text_from_pdf(pdf_file, max_pages=5):
  """Extracts text from the first N pages (typically Title, Abstract, Intro, Methods)."""
  reader = PdfReader(pdf_file)
  text = ""
  num_pages = min(len(reader.pages), max_pages)
  for page_num in range(num_pages):
    extracted = reader.pages[page_num].extract_text()
    if extracted:
      text += f"\n--- Page {page_num + 1} ---\n" + extracted
  return text


def evaluate_article(api_key, paper_text, boolean_str, inc_criteria, exc_criteria):
  """Calls OpenAI GPT model with structured output to screen the paper."""
  client = OpenAI(api_key=api_key)

  system_prompt = """
    You are an expert academic research assistant performing systematic literature review (SLR) screening.
    Your task is to evaluate an academic paper against a set of Inclusion Criteria, Exclusion Criteria, and a Boolean Search Scope.
    
    You must evaluate whether the paper should be IN SCOPE or OUT OF SCOPE.
    
    Be strict, objective, and reference specific criteria in your reasoning.
    """

  user_prompt = f"""
    ### SLR SEARCH SCOPE / BOOLEAN STRING:
    {boolean_str}

    ### INCLUSION CRITERIA:
    {inc_criteria}

    ### EXCLUSION CRITERIA:
    {exc_criteria}

    ### ARTICLE TEXT EXTRACT (Title, Abstract, Introduction):
    {paper_text[:8000]}  # Truncate to fit context cleanly

    ---
    Evaluate this paper and respond strictly in the following JSON format:
    {{
        "decision": "IN SCOPE" or "OUT OF SCOPE" or "UNCERTAIN",
        "confidence_score": 0.0 to 1.0,
        "summary": "Brief 2-sentence summary of the paper's core topic.",
        "matched_inclusion_criteria": ["List specific inclusion criteria met"],
        "violated_exclusion_criteria": ["List specific exclusion criteria triggered, if any"],
        "reasoning": "Detailed breakdown explaining why the paper is included or excluded based on the protocol."
    }}
    """

  response = client.chat.completions.create(
      model="gpt-4o-mini",  # Fast, cost-effective, and accurate for classification
      response_format={"type": "json_object"},
      messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt},
      ],
      temperature=0.1,  # Low temperature for deterministic output
  )

  return json.loads(response.choices[0].message.content)


# Action Trigger
if uploaded_file is not None:
  if not api_key:
    st.error("Please enter your OpenAI API Key in the sidebar to proceed.")
  else:
    if st.button("Run SLR Screening Analysis"):
      with st.spinner("Extracting text and screening paper against protocol..."):
        # 1. Extract text
        paper_text = extract_text_from_pdf(uploaded_file)

        # 2. Evaluate with LLM
        try:
          result = evaluate_article(
              api_key,
              paper_text,
              boolean_query,
              inclusion_criteria,
              exclusion_criteria,
          )

          # 3. Display Results
          st.divider()
          st.subheader("Screening Decision")

          decision = result.get("decision")
          if decision == "IN SCOPE":
            st.success(
                f"**Decision: IN SCOPE** (Confidence:"
                f" {result.get('confidence_score', 'N/A')})"
            )
          elif decision == "OUT OF SCOPE":
            st.error(
                f"**Decision: OUT OF SCOPE** (Confidence:"
                f" {result.get('confidence_score', 'N/A')})"
            )
          else:
            st.warning(
                f"**Decision: UNCERTAIN / MANUAL REVIEW REQUIRED** (Confidence:"
                f" {result.get('confidence_score', 'N/A')})"
            )

          col1, col2 = st.columns(2)

          with col1:
            st.markdown("### Article Summary")
            st.write(result.get("summary"))

            st.markdown("### Matched Inclusion Criteria")
            for inc in result.get("matched_inclusion_criteria", []):
              st.markdown(f"- ✅ {inc}")

          with col2:
            st.markdown("### Decision Rationale")
            st.write(result.get("reasoning"))

            st.markdown("### Triggered Exclusion Criteria")
            excs = result.get("violated_exclusion_criteria", [])
            if excs:
              for exc in excs:
                st.markdown(f"- ❌ {exc}")
            else:
              st.markdown("*No exclusion criteria triggered.*")

        except Exception as e:
          st.error(f"An error occurred during processing: {str(e)}")
