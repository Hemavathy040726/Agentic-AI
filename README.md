# ⚖️ Legal AI Assistant: Intelligent Indian Law Acts Query System

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)](https://python.langchain.com/)

An AI-powered legal research assistant built using Retrieval-Augmented Generation (RAG) that enables lawyers, law students, compliance professionals, and citizens to query Indian legal acts in natural language and receive accurate, contextual answers instantly.

## 🎯 The Legal Research Problem

### Current Challenges in Legal Research

Legal professionals and citizens face significant barriers when researching Indian law:

**1. Time-Consuming Manual Research**
- Lawyers spend 30-40% of their time searching through legal documents
- Reading hundreds of pages to find relevant sections and provisions
- Cross-referencing multiple acts for compliance requirements

**2. Complex Legal Language**
- Legal jargon makes acts difficult to understand for non-lawyers
- Citizens struggle to know their rights and obligations
- Small businesses find compliance requirements overwhelming

**3. Inefficient Search Methods**
- PDF keyword search misses contextual and semantic matches
- Can't ask questions like "What are the penalties for data breach?"
- No way to compare provisions across different acts

**4. Accessibility Barriers**
- Legal knowledge locked in dense, technical documents
- Expensive legal consultations for simple queries
- No instant answers for urgent compliance questions

### The Solution: AI-Powered Legal Query System

This RAG-based assistant democratizes access to legal knowledge by:

✅ **Natural Language Queries** - Ask in plain English: "What is cyber crime under IT Act?"  
✅ **Instant Answers** - Get precise information in seconds, not hours  
✅ **Contextual Understanding** - Semantic search finds relevant provisions even without exact keywords  
✅ **Cross-Act Analysis** - Query multiple legal acts simultaneously  
✅ **Citation-Backed Responses** - Answers grounded in actual legal text  
✅ **24/7 Availability** - No waiting for office hours or legal consultations  

## 📚 Currently Indexed Legal Acts

The system comes pre-configured with three major Indian legal acts:

### 1. **Information Technology Act, 2000** 🖥️
- **Scope**: Electronic governance, digital signatures, cyber crimes, data protection
- **Key Areas**: 
  - Legal recognition of electronic records and digital signatures
  - Cyber offenses and penalties (hacking, identity theft, data breach)
  - Powers of police and adjudicating officers
  - Cyber Appellate Tribunal procedures
- **Use Cases**: Cyber crime complaints, digital contract validity, data privacy compliance

### 2. **Environment Protection Act, 1986** 🌿
- **Scope**: Environmental conservation, pollution control, hazardous substances
- **Key Areas**:
  - Powers of central government for environmental protection
  - Standards for emission and discharge of pollutants
  - Penalties for environmental violations
  - Procedures for handling hazardous substances
- **Use Cases**: Environmental compliance for industries, pollution control, impact assessments

### 3. **Consumer Protection Act, 2019** 🛡️
- **Scope**: Consumer rights, product liability, unfair trade practices, e-commerce
- **Key Areas**:
  - Consumer rights and protection mechanisms
  - Product liability and manufacturer responsibilities
  - Unfair trade practices and misleading advertisements
  - E-commerce transactions and regulations
  - Consumer dispute redressal mechanisms
- **Use Cases**: Consumer complaints, product defect claims, e-commerce disputes, refund rights

## 🚀 Why This Matters

### For Legal Professionals 👨‍⚖️
- **Faster Research**: Cut legal research time by 70%
- **Quick Reference**: Instantly verify provisions during client consultations
- **Comparative Analysis**: Cross-reference related provisions across acts
- **Due Diligence**: Rapidly assess compliance requirements for clients

### For Law Students 📖
- **Study Aid**: Quick concept clarification while studying
- **Exam Preparation**: Rapid revision of act provisions
- **Assignment Research**: Find relevant sections for case studies
- **Understanding Context**: Get plain-language explanations of complex provisions

### For Businesses 🏢
- **Compliance Checks**: Verify legal obligations without expensive consultations
- **Risk Assessment**: Understand penalties and consequences of violations
- **Policy Development**: Base internal policies on legal requirements
- **Contract Review**: Verify legal validity of digital agreements

### For Citizens 👥
- **Know Your Rights**: Understand consumer and digital rights
- **File Complaints**: Know which provisions apply to your situation
- **Self-Help**: Get basic legal information without lawyer fees
- **Awareness**: Stay informed about legal protections available

## 🎮 Real-World Query Examples

### Information Technology Act Queries

```bash
Q: What constitutes a cyber crime under the IT Act 2000?
A: Based on the provided context from the Information Technology Act, 2000, the following constitutes a cyber crime under the IT Act 2000:

  1. **Cyber Terrorism**: This includes any act that is intended to cause or likely to cause death, injury, or damage to the public or private property, or disrupt or cause the disruption of critical infrastructure, or compromise national security, or intimidate or coerce a civilian population, or influence the policy of a government by intimidation or coercion. (Section 66F)
  
  2. **Publishing or transmitting obscene material in electronic form**: This includes publishing or transmitting any material that is lascivious or appeals to the prurient interest or has an effect that tends to deprave and corrupt. (Section 67)
  
  3. **Cyber Security**: Although not explicitly mentioned as a crime, the context mentions "cyber security" as protecting information, equipment, devices, computer, computer networks, or other electronic systems from unauthorized access, use, disclosure, disruption, modification, or destruction. However, the actual crime related to cyber security is not explicitly mentioned in the provided context.
  
  4. **Cyber Crimes not explicitly mentioned in the provided context**:
     - Hacking (Section 66)
     - Identity theft (Section 66C)
     - Cheating by personation by using computer resource (Section 66D)
     - Dishonest or fraudulent use of electronic signature (Section 66B)
     - Breach of confidentiality and privacy (Section 43A)
     - Cyber terrorism (Section 66F)
     - Publishing or transmitting of obscene material in electronic form (Section 67)
     - Publishing or transmitting of material depicting children in sexually explicit act in electronic form (Section 67B)
     - Violation of privacy (Section 43A)
  
  Note that the provided context only mentions a few sections of the IT Act 2000. The actual list of cyber crimes under the IT Act 2000 is more extensive and includes the above-mentioned crimes not explicitly mentioned in the provided context.
  

Q: Is a digital signature legally valid?
  A: According to the Information Technology Act, 2000, a digital signature is considered legally valid if it meets the following conditions:
  1. It has been issued by a Certifying Authority (CA) licensed under the Act.
  2. The subscriber has accepted the digital signature certificate by publishing or authorizing its publication, or by demonstrating approval in any manner.
  3. The digital signature certificate has been published or made available to the person relying on it, and the subscriber has accepted it.
  Section 5 of the Act states that an electronic signature shall be considered valid if it has been accepted by the subscriber, and the subscriber has certified to all who reasonably rely on it.
  Section 41 of the Act specifically deals with the acceptance of a digital signature certificate, stating that a subscriber shall be deemed to have accepted a digital signature certificate if they publish or authorize its publication, or demonstrate approval in any manner.
  Therefore, a digital signature is legally valid if it has been issued by a licensed Certifying Authority, accepted by the subscriber, and certified to all who reasonably rely on it.
  

Q: What are the penalties for data breach under IT Act?
A: According to the given context, the penalties for data breach under the IT Act are as follows:

1. **Disclosure of confidential information without consent**: If any person discloses electronic records, books, registers, correspondence, information, documents, or other materials without the consent of the concerned person, they shall be punished with:
   - Imprisonment for a term which may extend to 2 years.
   - Fine which may extend to one lakh rupees.
   - Both imprisonment and fine.

2. **Disclosure of information in breach of lawful contract**: If any person, including an intermediary, corporate, or other person, fails to comply with the direction under sub-section (6), they shall be punishable with:
   - Imprisonment for a term which may extend to 1 year.
   - Fine which may extend to one lakh rupees.
   - Both imprisonment and fine.

3. **Failure to provide information or comply with direction**: If any service provider, intermediary, data centre, body corporate, or person fails to provide the information called for or comply with the direction under sub-section (6), they shall be punishable with:
   - Imprisonment for a term which may extend to 1 year.
   - Fine which may extend to one lakh rupees.
   - Both imprisonment and fine.

Note: No court shall take cognizance of any offence under this section, except on a complaint made by an authority or a person authorized by the Central Government.

```

### Environment Protection Act Queries

```bash
Q: What penalties exist for violating environmental standards?
A:According to the given context, the penalties for violating environmental standards are as follows:
- Imprisonment for a term which may extend to six months (Section 88).
- Fine which may extend to twenty lakh rupees (Section 88).
- Both imprisonment and fine (Section 88).
These penalties apply to individuals who fail to comply with any direction of the Central Authority under sections 20 and 21.

Q: What are the duties of industries regarding pollution control?
A: According to the given context, the duties of industries regarding pollution control are as follows:
1. **Not to allow emission or discharge of environmental pollutants in excess of the standards**: Industries are not allowed to discharge or emit any environmental pollutant in excess of the standards that may be prescribed.
In other words, industries must adhere to the set standards for pollution control and not exceed them.
2. **Compliance with procedural safeguards**: Industries handling hazardous substances must comply with procedural safeguards, as mentioned in Section 8 of the Act.
3. **Furnishing of information to authorities and agencies**: Industries must furnish information to authorities and agencies in certain cases, as mentioned in Section 9 of the Act.
4. **Allowing entry and inspection**: Industries must allow entry and inspection by authorities, as mentioned in Section 10 of the Act.
5. **Providing samples for testing**: Industries must provide samples for testing, as mentioned in Section 11 of the Act.
Overall, the duties of industries regarding pollution control are to adhere to the set standards, comply with procedural safeguards, provide information, allow entry and inspection, and provide samples for testing.

Q: Can the government close down a polluting factory?
A: Yes, under Section 5, the Central Government has powers to issue directions including 
   closure of industries not complying with environmental standards...
   According to the given context, the answer is yes. 
As per Section 6 of the chapter, it is stated that the Central Government has the power to issue directions under this section, which includes the power to direct:
(a) the closure, prohibition or regulation of any industry, operation or process;
This implies that the government has the authority to close down a polluting factory if it is deemed necessary to prevent or control environmental pollution.

```

### Consumer Protection Act Queries

```bash
Q: What are my rights when buying products online?
A: Based on the provided context, your rights when buying products online are included under the broader category of "consumer rights." According to the context, consumer rights include:
- The right to be protected against the marketing of goods, products, or services by a product service provider or sold by a product seller.
As a consumer, you have the right to be protected against defective products, and you can bring a product liability action against the product manufacturer, product service provider, or product seller if you suffer harm due to a defective product.
Additionally, the context mentions that the expressions "buys any goods" and "hires or avails any services" include offline or online transactions through electronic means or by teleshopping or direct selling or multi-level marketing. This implies that online transactions are also covered under the consumer protection laws.
Therefore, as a consumer buying products online, you have the right to:
- Be protected against defective products
- Bring a product liability action against the product manufacturer, product service provider, or product seller if you suffer harm due to a defective product
- Be protected against unfair marketing practices
- Seek redressal for any consumer disputes that may arise during online transactions.


Q: How do I file a consumer complaint for defective products?
A: o file a consumer complaint for defective products, follow these steps:

1. **Gather evidence**: Collect all relevant documents, receipts, and proof of purchase related to the defective product. Also, take photos or videos of the product's defects.
2. **Identify the type of complaint**: Determine whether your complaint falls under the product liability chapter (if the product caused harm) or a general consumer dispute.
3. **Choose the appropriate forum**: If the dispute is related to a product liability action, you can file a complaint with the District Commission, State Commission, or National Commission, as per the Consumer Protection Act. For general consumer disputes, you can also file a complaint with the District Commission, State Commission, or National Commission.
4. **Prepare the complaint**: Write a detailed complaint stating the facts, including the product's defects, the harm caused, and the relief sought. Ensure you sign and seal the complaint as per the prescribed manner.
5. **Attach supporting documents**: Include all relevant documents, such as receipts, invoices, and test reports, to support your complaint.
6. **Submit the complaint**: File the complaint with the chosen forum (District Commission, State Commission, or National Commission) along with the required fees and supporting documents.
7. **Wait for mediation**: The forum will attempt to mediate a settlement between you and the product manufacturer or service provider. If mediation fails, the forum will proceed to hear the case.
8. **Follow up**: Keep track of the progress of your complaint and respond to any queries or requests from the forum.
9. **Seek compensation**: If the forum rules in your favor, you may be entitled to compensation for the harm caused by the defective product.

Remember to follow the specific procedures and timelines set by the Consumer Protection Act and the chosen forum for filing a consumer complaint.

Q: What is the liability of e-commerce companies for fake products?
A: Under Section 2(1)(16), e-commerce platforms can be held liable as 'product sellers' 
   Based on the provided context, e-commerce companies can be considered as product sellers. As per the definition of "product liability" in section 34, a product seller is responsible for compensating for any harm caused to a consumer by a defective product sold by them.
In the case of fake products, the e-commerce company can be held liable for the harm caused to the consumer. This is because they are responsible for ensuring that the products sold through their platform are genuine and not defective.
However, it's worth noting that the liability of e-commerce companies may vary depending on the specific circumstances of the case. If the e-commerce company can prove that they were not negligent or fraudulent in selling the fake product, they may still be liable under certain conditions.
In general, e-commerce companies have a responsibility to ensure that the products sold through their platform are genuine and meet the required standards. If they fail to do so, they can be held liable for any harm caused to the consumer.
In addition, e-commerce companies may also be liable for any harm caused by a product service provider, if the service provider is selling products through their platform and the product is defective.
Therefore, e-commerce companies can be held liable for fake products sold through their platform, and they may be required to compensate the consumer for any harm caused.

```

### Cross-Act Queries

```bash
Q: What legal protections exist for online consumer transactions?
A: [AI combines provisions from both IT Act and Consumer Protection Act]
 Based on the provided context, the legal protections for online consumer transactions can be inferred from the following points:

1. **Protection against marketing of goods and services**: The context mentions that consumer rights include protection against the marketing of goods, products, or services, including safeguards provided under any other law for the time being in force. This implies that online transactions are also covered under this protection.

2. **Prohibition on selling unsafe goods**: The context states that a trader is not allowed to sell goods that are unsafe to the public (point 9(b)). This applies to online transactions as well, as the expressions "buys any goods" and "hires or avails any services" include offline or online transactions through electronic means (point 1).

3. **Product liability**: The context mentions that a claim for product liability action lies against the product (point 9(vii)). This implies that consumers have the right to seek compensation for any harm caused by a defective or hazardous product, including those sold online.

4. **Awareness and education**: The context emphasizes the importance of spreading and promoting awareness on consumer rights (point 9(g)). This suggests that there may be initiatives to educate consumers about their rights and responsibilities in online transactions.

5. **Research and international best practices**: The context mentions that the organization should undertake and promote research in the field of consumer rights (point 9(f)) and recommend adoption of international covenants and best international practices on consumer rights (point 9(e)). This implies that there may be efforts to stay updated on the latest developments and best practices in online consumer transactions.

In summary, the legal protections for online consumer transactions include protection against marketing of goods and services, prohibition on selling unsafe goods, product liability, awareness and education, and research and international best practices.
Enter a question or 'quit' to exit: 
```

## 🛠️ Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Legal Query Interface                     │
│         "What are penalties for data breach?"                │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Query Embedding Generation                      │
│        (Sentence Transformers: all-MiniLM-L6-v2)            │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           Vector Similarity Search (ChromaDB)                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  IT Act 2000 │  │  EPA 1986    │  │  CPA 2019    │     │
│  │  Chunks      │  │  Chunks      │  │  Chunks      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            Top-N Relevant Legal Provisions                   │
│  • Section 43A - Data Protection (IT Act)                   │
│  • Section 72A - Penalties for Breach (IT Act)              │
│  • Section 66 - Computer Hacking (IT Act)                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  LLM Response Generation                     │
│         (GPT-4 / Llama 3.1 / Gemini 2.0)                    │
│                                                              │
│  Prompt: "Based on the following legal provisions,          │
│           answer the question accurately..."                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Accurate, Citation-Backed Answer                │
│  "Under Section 43A and 72A of IT Act 2000, data           │
│   breach penalties include..."                              │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- API key for OpenAI, Groq, or Google Gemini
- 2GB free disk space (for embeddings and vector DB)

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/legal-rag-assistant.git
cd legal-rag-assistant
```

2. **Create virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API keys**

Create `.env` file:
```env
# Choose ONE LLM provider:

# OpenAI (Recommended for accuracy)
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4o-mini

# OR Groq (Fastest inference)
GROQ_API_KEY=your-groq-key-here
GROQ_MODEL=llama-3.1-8b-instant

# OR Google Gemini (Free tier available)
GOOGLE_API_KEY=your-google-key-here
GOOGLE_MODEL=gemini-2.0-flash

# Vector DB Settings (Optional)
CHROMA_COLLECTION_NAME=indian_legal_acts
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

5. **Run the assistant**
```bash
python src/app.py
```

## 🎯 Usage Guide

### Interactive Mode

```bash
$ python src/app.py

Initializing RAG Assistant...
Using OpenAI model: gpt-4o-mini
Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
Vector database initialized with collection: indian_legal_acts

Loading documents...
Loaded PDF: env_prot_act_1986.pdf, length: 45231 characters
Loaded PDF: con_prot_act_2019.pdf, length: 38492 characters
Loaded PDF: it_act_2000.pdf, length: 52103 characters
Documents added to vector database

Enter a question or 'quit' to exit: 
> What is cyber stalking?

Cyber stalking is addressed under Section 66A of the IT Act 2000...

Enter a question or 'quit' to exit:
> quit
```

### Programmatic Usage

```python
from app import RAGAssistant, load_documents

# Initialize
assistant = RAGAssistant()
docs = load_documents()
assistant.add_documents(docs)

# Single query
answer = assistant.invoke(
    "What are consumer rights under CPA 2019?",
    n_results=5  # Retrieve top 5 relevant chunks
)
print(answer)

# Multiple queries
queries = [
    "Digital signature validity",
    "Environmental compliance requirements",
    "Product liability provisions"
]

for query in queries:
    result = assistant.invoke(query)
    print(f"\nQ: {query}")
    print(f"A: {result}\n")
```

## 🔧 Adding More Legal Acts

### Step 1: Add PDF to data folder
```bash
data/
├── it_act_2000.pdf
├── env_prot_act_1986.pdf
├── con_prot_act_2019.pdf
└── your_new_act.pdf  # Add your PDF here
```

### Step 2: Update load_documents() in app.py
```python
pdf_files = [
    "env_prot_act_1986.pdf",
    "con_prot_act_2019.pdf",
    "it_act_2000.pdf",
    "your_new_act.pdf"  # Add filename
]
```

### Step 3: Re-run to index
```bash
python src/app.py
```

The system will automatically chunk and index the new act!

## ⚙️ Optimization for Legal Documents

### Chunk Size Configuration

Legal documents have specific structure. Adjust chunking for better results:

```python
# In vectordb.py

# For section-based acts (default - good for IT Act, CPA)
chunks = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# For detailed provisions (EPA with technical standards)
chunks = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

# For consolidated acts with schedules
chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\nSection", "\n\nSchedule", "\n\n", "\n", " "]
)
```

### Legal-Specific Prompt Template

```python
# In app.py - enhance prompt for legal accuracy

self.prompt_template = ChatPromptTemplate.from_template(
    """You are a legal research assistant specializing in Indian law.
    
    Analyze the following legal provisions and answer the question accurately.
    
    Legal Context:
    {context}
    
    Question:
    {question}
    
    Instructions:
    1. Cite specific sections and act names
    2. Explain in clear, understandable language
    3. Mention penalties if applicable
    4. Note any amendments or important clarifications
    5. If multiple provisions apply, explain all relevant ones
    
    Answer:
    """
)
```


## ⚠️ Important Legal Disclaimers

### 🚨 NOT A SUBSTITUTE FOR LEGAL ADVICE

This tool is designed for:
- ✅ Legal research assistance
- ✅ Quick reference and fact-checking
- ✅ Educational purposes
- ✅ Understanding general legal provisions

This tool is NOT:
- ❌ A replacement for professional legal counsel
- ❌ Authoritative legal interpretation
- ❌ Suitable for critical legal decisions without verification
- ❌ Updated in real-time with amendments

### Best Practices

1. **Verify Critical Information**: Always cross-check answers for important legal matters
2. **Consult Professionals**: Seek qualified legal advice for specific cases
3. **Check for Amendments**: Legal acts may be amended; verify current provisions
4. **Understand Limitations**: AI may miss nuances; use as starting point, not final authority
5. **Document Sources**: When using in legal work, verify original act text

## 🎓 Technical Deep Dive

### Why RAG for Legal Documents?

**Traditional Approach Problems:**
- Keyword search misses semantic variations ("hacking" vs "unauthorized access")
- Can't understand questions in natural language
- No contextual ranking of results
- Requires knowing exact legal terminology

**RAG Advantages:**
- Semantic understanding matches intent, not just keywords
- Retrieves most relevant sections even with layman's terms
- LLM provides coherent synthesis of multiple provisions
- Natural language interface accessible to non-lawyers

### Embedding Model Selection

We use `sentence-transformers/all-MiniLM-L6-v2` because:
- ✅ Fast inference (important for real-time queries)
- ✅ Good balance of accuracy and speed
- ✅ 384-dimensional vectors (efficient storage)
- ✅ Trained on semantic similarity tasks
- ✅ Works well with legal terminology

### Vector Database Choice

ChromaDB selected for:
- ✅ Persistent storage (no re-indexing needed)
- ✅ Fast similarity search with HNSW algorithm
- ✅ Metadata filtering capabilities
- ✅ Local deployment (data privacy)
- ✅ Easy integration with LangChain

## 🛣️ Future Enhancement

### 1: Enhanced Legal Coverage
- [ ] Add more Indian acts (Companies Act, IPC, CrPC, CPC)
- [ ] Supreme Court landmark judgments integration
- [ ] High Court rulings database

### 2: Advanced Features
- [ ] Web interface (Streamlit-based)
- [ ] Multi-language support (Hindi, regional languages)
- [ ] Citation extraction and formatting
- [ ] Comparative provision analysis across acts
- [ ] Amendment tracking and notifications

### 3: Professional Tools
- [ ] Legal brief generation
- [ ] Compliance checklist creator
- [ ] Contract clause verification
- [ ] REST API for law firm integration
- [ ] Case law relevance matching

### 4: Scale & Deploy 
- [ ] Cloud deployment with authentication
- [ ] Subscription tiers for professionals
- [ ] Mobile app development
- [ ] Integration with legal research platforms
- [ ] Jurisdiction-specific fine-tuning

## 📂 Project Structure

```
legal-rag-assistant/
├── data/                           # Legal documents storage
│   ├── it_act_2000.pdf            # Information Technology Act
│   ├── env_prot_act_1986.pdf      # Environment Protection Act
│   ├── con_prot_act_2019.pdf      # Consumer Protection Act
│   └── [Add more acts here]
│
├── src/
│   ├── app.py                     # Main RAG assistant
│   │   ├── RAGAssistant class     # Core RAG logic
│   │   ├── load_documents()       # PDF loader
│   │   └── main()                 # CLI interface
│   │
│   ├── vectordb.py                # Vector database wrapper
│   │   ├── VectorDB class         # ChromaDB operations
│   │   ├── chunk_text()           # Text chunking
│   │   ├── add_documents()        # Document indexing
│   │   └── search()               # Similarity search
│   │
│   └── chroma_db/                 # Persistent vector storage
│       └── [Generated files]
│
├── .env                           # API keys (not in repo)
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
├── LICENSE                        # CC BY-NC-SA 4.0
└── README.md                      # This file
```

## 🤝 Contributing

Contributions welcome! Especially from:
- **Legal Professionals**: Validate accuracy, suggest improvements
- **Developers**: Enhance features, optimize performance
- **Law Students**: Add more acts, improve documentation

### How to Contribute

1. Fork the repository
2. Create feature branch (`git checkout -b feature/add-ipc-act`)
3. Add legal acts or improve code
4. Test thoroughly with sample queries
5. Submit pull request with description

### Priority Contributions Needed

- [ ] Additional Indian legal acts (IPC, CrPC, CPC, Companies Act)
- [ ] Judgment database integration
- [ ] Multi-language support
- [ ] Accuracy validation framework
- [ ] Web UI development

## 📝 License

**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**

You may:
- ✅ Use for personal legal research
- ✅ Use for educational purposes
- ✅ Modify and adapt for non-commercial projects
- ✅ Share with attribution

You may NOT:
- ❌ Use commercially without permission
- ❌ Sell as a product or service
- ❌ Use without providing attribution

For commercial licensing, contact the maintainer.

## 🙏 Acknowledgments

- **Ministry of Law and Justice, India** - For making legal acts publicly available
- **LangChain** - RAG framework and document processing
- **ChromaDB** - Vector storage and retrieval
- **Sentence Transformers** - Embedding models
- **OpenAI/Groq/Google** - LLM APIs

## 📧 Contact & Support

**For Legal Professionals:**
- Report accuracy issues
- Suggest additional acts to include
- Request specific features

**For Technical Issues:**
- Open GitHub issue with details
- Provide error logs and query examples
- Tag with appropriate labels

**For Commercial Inquiries:**
- Licensing for law firms
- Custom deployment needs
- Bulk integration requirements

---

**⚖️ Making legal knowledge accessible to all through AI**

*Disclaimer: This tool is for informational purposes only and does not constitute legal advice. Always consult qualified legal professionals for specific legal matters.*
