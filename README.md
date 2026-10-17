# NLP Assignment 4 — Authentic RAG vs Improved RAG

**ผู้จัดทำ:** รหัสนักศึกษา `68056030`  
**อ้างอิงต้นฉบับ:** [aekanun2020/2025-authenticRAG](https://github.com/aekanun2020/2025-authenticRAG)  
**สภาพแวดล้อมทดสอบ:** Windows, Conda Python 3.10, Docker Desktop, OpenSearch 2.19.1  

## เป้าหมาย
โจทย์ข้อ (1)–(3): นำ Authentic RAG จาก Lab ของอาจารย์มาทดลอง แล้วปรับปรุง Retrieval โดย **ไม่เปลี่ยน LLM สำหรับสร้างคำตอบสุดท้าย** ใช้ Qwen `qwen/qwen-2.5-72b-instruct` ผ่าน OpenRouter เหมือนกันทั้งสองระบบ

## ไฟล์หลัก

| ไฟล์ | หน้าที่ |
|---|---|
| `authenticRAG.py` | Baseline: โหลด Markdown, Chunking, ทำ Embedding + Indexing |
| `onlysearchAuthenticRAG.py` | Baseline: ค้นหา BM25 + Vector + RRF และสร้างคำตอบ |
| `improvedRAG.py` | Improved: MarkdownNodeParser + SentenceSplitter (512 tokens, overlap 80), ทำ Embedding + Indexing |
| `onlysearchimprovedRAG.py` | Improved: Hybrid RRF → Top-20 → BGE Reranker v2 M3 → Top-5 → Qwen |
| `COMPARISON_REPORT.md` | อธิบายความต่าง ผลดีและข้อจำกัด พร้อมผลทดลองเชิงคุณภาพ |
| `retrieval_summary.csv` | ภาพรวม Top-1 และจำนวนผลจากคำถาม 5 ข้อ |
| `review_labels.csv` | ตารางให้ผู้ประเมินทำ relevance judgment ทีละ Chunk |
| `evaluate.py` | คำนวณ Hit@5, Precision@5, MRR@5 และ nDCG@5 เฉพาะกรณีที่มีฉลากครบ |
| `evaluation_metrics.csv` | ไฟล์สรุป Metrics (สร้างโดย evaluate.py) |

`corpus_input/` ต้องมีไฟล์ `1.md`, `2.md`, `44.md`, `5555.md` ตรงกับการทดลองเดิม (ยังไม่ได้รวมไฟล์ corpus ต้นฉบับไว้ในแพ็กเกจนี้)

## 1) Setup บน Windows (Anaconda Prompt)

```bat
conda create -n advrag python=3.10 -y
conda activate advrag
python -m pip install -r requirements-windows.txt
python -m pip check
```

เปิด Docker Desktop และรัน OpenSearch ตามคู่มืออาจารย์ (ครั้งแรกเท่านั้น):

```bat
docker network create opensearch-net
docker run -d --name opensearch -p 127.0.0.1:9200:9200 -p 127.0.0.1:9600:9600 -e "discovery.type=single-node" -e "DISABLE_SECURITY_PLUGIN=true" -e "OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g" opensearchproject/opensearch:2.19.1
```

หากมี container แล้วให้ใช้ `docker start opensearch` ไม่ต้องสร้างซ้ำ แล้วตรวจสอบด้วย `curl http://localhost:9200` ระบบทดลองนี้ปิด Security Plugin ใช้บน localhost เท่านั้น

สร้าง OpenRouter API key **ใหม่และเก็บเป็นความลับ** (ห้าม commit key) แล้วตั้งใน CMD session ด้วย `set OPENROUTER_API_KEY=YOUR_NEW_API_KEY` ก่อนรัน Indexing/Search; การตั้งค่าชั่วคราวนี้จะหายเมื่อปิดหน้าต่าง

## 2) รัน Baseline และ Improved

```bat
cd /d C:\Users\kumin\Desktop\NLP_RAG_68056030
python authenticRAG.py
python onlysearchAuthenticRAG.py
python improvedRAG.py
python onlysearchimprovedRAG.py
```

**หมายเหตุ:** ผมปรับชื่อ output ใน `onlysearchimprovedRAG.py` เป็น `improved_rag_search_results.json` แล้ว เพื่อไม่เขียนทับ Baseline ทั้งสอง JSON ในชุดนี้เป็นผลทดลองจริง ไม่จำเป็นต้อง Index ใหม่ (การ Indexing ซ้ำอาจมีค่า API)

## 3) เทียบผลทดลอง (จาก JSON ที่ผู้ทดลองบันทึกจริง)

| ประเด็น | Baseline | Improved |
|---|---|---|
| Corpus | 4 Markdown files | 4 Markdown files |
| Chunks | 53 | 203 |
| Index | `anthropic-vector-index`, `anthropic-bm25-index` | ชื่อเหมือนเดิมแต่มี `-improved` ต่อท้าย |
| Embedding | `BAAI/bge-m3` | `BAAI/bge-m3` |
| Retrieval | BM25 + Dense + RRF → Top-5 | BM25 + Dense + RRF → 20 candidates |
| Reranking | ไม่มี | `BAAI/bge-reranker-v2-m3` → Top-5 |
| Final LLM | Qwen 2.5 72B | Qwen 2.5 72B |

### ข้อค้นพบตาม 5 คำถาม

1. **หัด vs หัดเยอรมัน:** ทั้งคู่พบข้อมูลภาพรวม แต่ Improved มี Chunk นอกประเด็นบางส่วนอยู่ใน Top-5
2. **สาเหตุและการป้องกัน:** Improved พบเอกสารสาเหตุอันดับ 1 และเอกสารป้องกันอันดับ 2; Baseline พบเอกสารป้องกันอันดับ 4
3. **หญิงตั้งครรภ์:** Baseline วางหัวข้อภาวะแทรกซ้อนตรงคำถามอันดับ 1; Improved วางอันดับ 2
4. **ฉีดวัคซีนแล้วยังติดเชื้อได้ไหม:** Improved พบข้อความตอบการติดเชื้อซ้ำโดยตรงอันดับ 1; Baseline พบเนื้อหาเกี่ยวกับวัคซีนในภาพรวมอันดับ 1
5. **ผลกระทบสาธารณสุขและเศรษฐกิจ:** ทั้งคู่พบหลักฐานเกี่ยวข้องทางอ้อม ยังไม่พิสูจน์ว่า Improved ดีขึ้น

**ข้อสรุป:** มีสัญญาณว่าจัดอันดับได้ดีขึ้นบางคำถาม โดยเฉพาะข้อ 2 และ 4 แต่ยังไม่สามารถสรุปเปอร์เซ็นต์ Accuracy หรือ Recall@5 ได้จากชุดผลนี้เพียงอย่างเดียว เพราะยังไม่มี Ground Truth ตรวจสอบอย่างเป็นระบบ คะแนน RRF และ Reranker ใช้คนละสเกลจึงเปรียบเทียบตัวเลขดิบข้ามระบบไม่ได้ การเปลี่ยนทั้ง Chunking และ Reranking พร้อมกันยังแยกผลของแต่ละเทคนิคไม่ได้

**ข้อจำกัดที่ต้องพูดให้ชัด:** Baseline Log แสดง Sparse/BM25 พบ 0 hits สำหรับ 5 คำถาม จึงเป็นการผสานเพียง Dense hits ในการทดลองนี้ อาจต้องทดลอง Thai tokenizer เพิ่มในอนาคต และคำตอบ LLM บางจุดมีข้อผิดพลาด เช่น Improved อ้างว่า MMR ป้องกันอีสุกอีใส ซึ่งไม่ถูกต้อง ต้องประเมิน factual support แยกจาก retrieval

## 4) Evaluation — ให้คะแนนอย่างตรวจสอบได้

ไฟล์ `review_labels.csv` มี 50 แถว = 2 ระบบ × 5 คำถาม × 5 Chunks เปิดด้วย Excel หรือ Text Editor แล้วกรอกคอลัมน์ `relevance_0_1_2` ให้ครบ: `0=ไม่เกี่ยวข้อง`, `1=เกี่ยวข้องบางส่วน`, `2=ตอบตรงประเด็น` ใช้ `content_preview` ช่วยดู แต่ **ควรเปิด JSON อ่าน Chunk เต็มก่อนให้คะแนน** และหากเป็นไปได้ ให้คนอื่นตรวจซ้ำ

```bat
python evaluate.py
```

จะได้ `evaluation_metrics.csv`; หากยังไม่ได้กำหนดฉลากครบ โปรแกรม **จะไม่รายงานคะแนนเฉลี่ยเหมือนวัดเสร็จแล้ว**

Metrics:
- Hit@5: มีผลอย่างน้อยหนึ่ง Chunk ที่ relevance >= 1 หรือไม่
- Precision@5: สัดส่วนเอกสาร 5 อันดับแรกที่ relevance >= 1
- MRR@5: ค่า reciprocal rank ของ Chunk ที่เกี่ยวข้องตัวแรก
- nDCG@5 (local): วัดความเรียงลำดับจากคะแนน 0/1/2 ภายในผล Top-5 ที่ตรวจแล้ว ไม่ใช่ corpus-wide nDCG
- **ไม่รายงาน Recall@5** เพราะไม่มีรายการ Chunk ที่เกี่ยวข้องทั้งหมดใน OpenSearch

## 5) วิธีนำงานขึ้น GitHub

ก่อน push ตรวจ `.gitignore` และไฟล์ที่จะอัปโหลด โดยเฉพาะ key, ไฟล์ `.env`, และเอกสารที่ไม่มีสิทธิ์เผยแพร่ ตรวจชื่อเจ้าของและ license ของโค้ดจากอาจารย์ด้วย

```bat
git init
git add .
git status
git commit -m "NLP Assignment 4: baseline and improved RAG"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/NLP_RAG_68056030.git
git push -u origin main
```

สร้าง Repository ว่างบน GitHub ก่อน และเปลี่ยน `YOUR_USERNAME` เป็นบัญชีของคุณ ตรวจสอบการมีสิทธิ์ใช้/เผยแพร่ Corpus ก่อน push

## 6) ส่งอาจารย์และอธิบายหน้าห้อง

ส่งลิงก์ GitHub ทาง Email โดยใช้ Subject: `KDAI-NLP-68056030-Assignment-4Oct2026` ก่อนเริ่มเรียนครั้งถัดไป

เตรียมอธิบาย: Dataset 4 ไฟล์ → Index Baseline 53 ชิ้น / Improved 203 ชิ้น → BM25 + Vector + RRF → Cross-Encoder เพิ่มใน Improved → ผลรายคำถามและข้อจำกัด → ไม่เปลี่ยน Qwen

เครดิต: โค้ด Baseline ดัดแปลงจาก Repository ของอาจารย์เอ้ [2025-authenticRAG](https://github.com/aekanun2020/2025-authenticRAG) สำหรับใช้ในการเรียน
