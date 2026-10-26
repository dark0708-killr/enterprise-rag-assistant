# Enterprise Master Software-as-a-Service (SaaS) Agreement & SLA
**Master Agreement No.:** MSA-ENT-2024-901  
**Governing Law:** State of Delaware, United States  
**Effective Date:** Binding upon Order Form execution  

---

## 1. Service Level Agreement (SLA) & Uptime Commitments
Provider guarantees a Monthly Uptime Percentage of **99.99%** ("Commitment Level"), calculated across all operational calendar hours, excluding scheduled maintenance windows announced at least 72 hours in advance.

### 1.1 Service Credits Table
In the event Provider fails to meet the Guaranteed Uptime Percentage, Customer is entitled to the following non-refundable Service Credits applicable against future invoice cycles:

| Monthly Uptime Percentage | Service Credit Percentage |
| :--- | :--- |
| **99.90% to < 99.99%** | **10% of monthly fees** |
| **99.00% to < 99.90%** | **25% of monthly fees** |
| **< 99.00%** | **50% of monthly fees** |

*Service Credit Claim Process:* Customer must submit written notice of claim within thirty (30) days following the calendar month in which the downtime incident occurred.

---

## 2. Data Protection, Isolation & Residency

### 2.1 Regional Data Sovereignty
* Provider provides isolated tenant infrastructure in designated geographic regions:
  * **US East (N. Virginia, AWS us-east-1)**
  * **EU Central (Frankfurt, AWS eu-central-1 - GDPR compliant)**
  * **Asia Pacific (Singapore, AWS ap-southeast-1)**
* Customer data, including vectorized embeddings and vector search indexes, shall never leave the selected sovereign data enclave without explicit written consent.

### 2.2 Artificial Intelligence & Model Training Safeguards
* **Zero Customer Data Retention Policy for Model Training:** Provider represents and covenants that **neither Customer Content, uploaded documents, nor Customer Queries/Prompts shall be utilized to train, fine-tune, or retrain any public or multi-tenant foundational AI models**.
* All embedding vector generation and model completions execute within dedicated, isolated inference endpoints.

---

## 3. Limitation of Liability
* **General Cap:** Except for breaches of Section 2 (Data Protection & Confidentiality) and gross negligence or willful misconduct, each party’s aggregate cumulative liability under this Agreement shall be limited to the total fees paid or payable by Customer in the twelve (12) months preceding the event giving rise to liability.
* **Consequential Damages Waiver:** Neither party shall be liable for indirect, incidental, punitive, or loss of prospective profits or business opportunities.

---

## 4. Term, Termination & Data Return
* **Term:** Initial term of twelve (12) months, renewing automatically for successive 12-month periods unless notice of non-renewal is provided at least sixty (60) days prior to expiration.
* **Data Return Upon Termination:** Provider shall provide Customer with an automated cryptographic export of all Customer Data in standard formats (JSON, Parquet, or SQL dump) within thirty (30) days of contract termination, followed by verified cryptographic DoD 5220.22-M sanitization across all media.
