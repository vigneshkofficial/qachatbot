Q&A DOCUMENT CHATBOT 

My project aims to address this need by extracting and processing relevant drug information from multiple PDFs. The system is designed to provide accurate answers
to user queries based on the extracted content and manage frequent inquiries by streamlining them into FAQs. 
The main challenges include ensuring precise text extraction and contextually relevant information retrieval.

To create a Q&A document chatbot using the Gemini API, start by setting up your environment with the necessary libraries like Streamlit, requests, 
PyPDF2, and ChromaDB. After obtaining your Gemini API key, allow users to upload medical documents in formats like PDF or DOCX. Use PyPDF2 to extract text 
from the uploaded documents. Once the text is extracted, send it to the Gemini API, which processes the text and allows the chatbot to answer user questions 
by querying the content.The answers are then returned to the user through the Streamlit interface, enabling a seamless interaction with the medical documents.

I have utilized the Gemini API to map user queries to the document content effectively, ensuring accurate and relevant responses
