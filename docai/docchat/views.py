import os
from django.conf import settings
from django.shortcuts import render
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from .forms import DocumentUploadForm, QuestionForm

def home(request):
    print("POST keys:", request.POST.keys())
    print("FILES:", request.FILES.keys())
    upload_form = DocumentUploadForm()
    question_form = QuestionForm()
    answer = None

    filename = request.session.get("uploaded_filename")

    if request.method == "POST" and "upload_file" in request.POST:
        upload_form = DocumentUploadForm(request.POST, request.FILES)
        print("POST keys:", request.POST.keys())
        print("FILES:", request.FILES.keys())
        if upload_form.is_valid():
            uploaded_file = upload_form.cleaned_data["file"]
            filename = uploaded_file.name

            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            save_path = settings.MEDIA_ROOT / filename
            with open(save_path, "wb") as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)

            request.session["conversation"] = []
            request.session["uploaded_filename"] = filename
            request.session.pop("vectorstore_path", None)  # optional, if you were storing
            request.session.modified = True

            loader = PyPDFLoader(str(save_path))
            documents = loader.load()

            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
            vectorstore = Chroma.from_documents(texts, embeddings)

            
            request.session["uploaded_filename"] = filename
            request.session["conversation"] = []
            request.session.modified = True

            return render(request, "docchat/upload_success.html", {
                "filename": filename,
                "question_form": question_form,
                "conversation": [],
                "answer": None
            })
   
    elif request.method == "POST" and "ask_question" in request.POST:
        question_form = QuestionForm(request.POST)
        if not filename:    
            return render(request, "docchat/upload.html", {
                "form": upload_form,
                "error": "Please upload a document first."
            })

        save_path = settings.MEDIA_ROOT / request.session["uploaded_filename"]
        loader = PyPDFLoader(str(save_path))
        documents = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
        vectorstore = Chroma.from_documents(texts, embeddings)

        if question_form.is_valid():
            question = question_form.cleaned_data["question"]
            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(temperature=0, openai_api_key=settings.OPENAI_API_KEY),
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )
            answer = qa.run(question)

            conversation = request.session.get("conversation", [])
            conversation.append({"question": question, "answer": answer})
            request.session["conversation"] = conversation
            request.session.modified = True

        return render(request, "docchat/upload_success.html", {
            "filename": filename,
            "question_form": QuestionForm(),
            "conversation": request.session.get("conversation", []),
        })
    
    elif request.method == "POST" and "reset" in request.POST:
        for key in ["conversation", "uploaded_filename", "vectorstore_path"]:
            request.session.pop(key, None)
        request.session.modified = True
        return render(request, "docchat/upload.html", {"form": upload_form})

    elif request.method == "POST" and "clear_chat" in request.POST:
        request.session["conversation"] = []
        request.session.modified = True

        return render(request, "docchat/upload_success.html", {
            "filename": request.session.get("uploaded_filename"),
            "question_form": QuestionForm(),
            "conversation": [],
            "answer": None
        })
    
    return render(request, "docchat/upload.html", {"form": upload_form})
