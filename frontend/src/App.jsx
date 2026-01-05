import React, { useState } from 'react';
import UploadReport from './components/UploadReport';
import ChatInterface from './components/ChatInterface';
import './App.css';

function App() {
  const [uploadedFileId, setUploadedFileId] = useState(null);
  const [extractedText, setExtractedText] = useState('');
  const [activeTab, setActiveTab] = useState('upload');

  const handleUploadSuccess = (fileId, text) => {
    console.log('📤 Upload Success - FileID:', fileId);
    console.log('📝 Extracted Text Length:', text?.length);
    setUploadedFileId(fileId);
    setExtractedText(text);
    setActiveTab('chat');
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🏥 Medical Report AI</h1>
        <p>Upload your medical reports and get instant insights</p>
      </header>

      <div className="tab-navigation">
        <button
          className={`tab-button ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTab('upload')}
        >
          📤 Upload Report
        </button>
        <button
          className={`tab-button ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => setActiveTab('chat')}
          disabled={!uploadedFileId}
        >
          💬 Ask Questions
        </button>
      </div>

      <main className="app-main">
        {activeTab === 'upload' && (
          <UploadReport onUploadSuccess={handleUploadSuccess} />
        )}

        {activeTab === 'chat' && uploadedFileId && (
          <ChatInterface
            fileId={uploadedFileId}
            reportText={extractedText}
          />
        )}

        {activeTab === 'chat' && !uploadedFileId && (
          <div className="info-message">
            <p>Please upload a report first to start asking questions.</p>
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>⚕️ Medical Report AI - For educational purposes only</p>
        <p>Always consult with healthcare professionals for medical advice</p>
      </footer>
    </div>
  );
}

export default App;
