# Vis-Assist ChatBot for Teachers
LLM based tool to assist visually impaired teachers analyze and tailor content.

## Objective
Empower visually impaired teachers in special schools by providing a Generative AI-powered tool to access and tailor educational content.
Visually impaired teachers often lack the tools and materials needed for effective teaching. They face challenges in finding accessible content and have limited learning opportunities compared to their peers in mainstream schools.

## Proposed Solution
Our solution is a Learning Module application powered by Generative AI, enabling visually impaired teachers to:

1) Upload documents or text materials.
2) Generate structured learning modules through AI processing.
3) Interact with the system via audio for content refinement.

## Technologies Used
-> LLM: Mixtral 8x7B (Mixture-of-Experts technique)

-> ASR: Whisper Large v3 (Automatic Speech Recognition) 

-> TTS: XTTS v2 (Text-to-Speech) 

-> Frameworks: Streamlit, LangChain 

## How the Models are working
1) Model Loading: The code loads various AI models for language understanding and generation, such as retrieval QA models and language models like Ollama.
2) Document Processing: When a PDF is uploaded, the code analyzes its content using text processing techniques like text splitting to prepare it for further interaction.
3) Interactive Chatbot: It employs a chatbot interface where users can interact via text or audio. The chatbot generates responses using the loaded language models.
4) Speech Synthesis: The generated responses are converted into speech using a text-to-speech (TTS) model, allowing users to hear the bot's replies.
5) PDF Generation: Tailored content created is then structured and converted into a PDF format for download.

## Conclusion

Our Generative AI-powered Vis-Assist ChatBot significantly enhances the teaching experience for visually impaired educators, providing them with tailored, accessible educational content. This project aims to bridge the gap between special and mainstream education, promoting inclusivity and accessibility.


