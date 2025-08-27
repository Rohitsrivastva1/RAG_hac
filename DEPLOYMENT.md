# 🚀 RAG System Deployment Guide - Render

## Overview
This guide will help you deploy your RAG system to Render without using Docker.

## Prerequisites
- A Render account (free tier available)
- Your RAG system code in a Git repository
- Gemini API key configured

## Deployment Steps

### 1. **Prepare Your Repository**
Ensure your repository contains these files:
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python version specification
- `Procfile` - Process definition for Render
- `render.yaml` - Blueprint configuration (optional)

### 2. **Deploy to Render**

#### Option A: Manual Deployment
1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Web Service"
3. Connect your Git repository
4. Configure the service:
   - **Name**: `rag-system`
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Starter` (free tier)

#### Option B: Blueprint Deployment (Recommended)
1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Blueprint"
3. Connect your Git repository
4. Render will automatically detect `render.yaml` and configure everything

### 3. **Environment Variables**
Set these in Render dashboard:
```
LLM_PROVIDER=gemini
LLM_MODEL=gemini-1.5-flash
GEMINI_API_KEY=your_actual_gemini_api_key
VECTOR_STORE_TYPE=faiss
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K=5
SIMILARITY_THRESHOLD=0.7
ENABLE_OCR=true
ENABLE_METRICS=false
DEBUG=false
```

### 4. **Post-Deployment**
- Your app will be available at: `https://your-app-name.onrender.com`
- The health check endpoint: `/health`
- API documentation: `/docs`

## Important Notes

### ⚠️ **Free Tier Limitations**
- **Sleep after inactivity**: Your app will sleep after 15 minutes of inactivity
- **Cold starts**: First request after sleep may take 30-60 seconds
- **Build time**: Limited to 45 minutes
- **Bandwidth**: 750 GB/month

### 🔧 **Production Considerations**
- **Vector Storage**: FAISS index is in-memory and will reset on each restart
- **Document Persistence**: Uploaded documents are stored temporarily
- **Scaling**: Consider upgrading to paid plans for production use

### 📁 **File Structure for Render**
```
RAG_hac/
├── app/
│   ├── main.py
│   ├── core/
│   └── services/
├── requirements.txt
├── runtime.txt
├── Procfile
├── render.yaml
└── DEPLOYMENT.md
```

## Troubleshooting

### Common Issues:
1. **Build Failures**: Check `requirements.txt` for compatibility
2. **Import Errors**: Ensure all dependencies are in `requirements.txt`
3. **Port Issues**: Render automatically sets `$PORT` environment variable
4. **Memory Issues**: Free tier has 512MB RAM limit

### Debug Commands:
- Check logs in Render dashboard
- Use `/health` endpoint to verify service status
- Check build logs for dependency issues

## Support
- Render Documentation: [docs.render.com](https://docs.render.com/)
- Render Community: [community.render.com](https://community.render.com/)
