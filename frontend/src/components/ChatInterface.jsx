import React, { useState, useRef, useEffect } from 'react';
import { queryReport } from '../services/api';
import './ChatInterface.css';

function ChatInterface({ fileId, reportText }) {
  console.log('💬 ChatInterface - FileID:', fileId);
  console.log('📄 ChatInterface - ReportText Length:', reportText?.length);
  
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Add welcome message with file confirmation
    if (fileId !== null && fileId !== undefined && fileId !== '') {
      const textLength = reportText ? reportText.length : 0;
      setMessages([
        {
          type: 'bot',
          text: `✅ Report uploaded successfully!\n\n📄 **Report ID:** ${fileId}\n📝 **Text extracted:** ${textLength} characters\n\n🩺 I've analyzed your medical report. Ask me questions like:\n• "What is my WBC count?"\n• "Analyze my complete blood report"\n• "What does high cholesterol mean?"\n• "Show me all my results"`,
          timestamp: new Date()
        }
      ]);
    } else {
      setMessages([
        {
          type: 'bot',
          text: '⚠️ No report uploaded yet. Please go to the "Upload Report" tab and upload your blood report first.',
          timestamp: new Date()
        }
      ]);
    }
  }, [fileId, reportText]);

  useEffect(() => {
    // Scroll to bottom when new messages arrive
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputText.trim() || loading) return;

    const userMessage = {
      type: 'user',
      text: inputText,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setLoading(true);

    try {
      const response = await queryReport(inputText, fileId);

      const botMessage = {
        type: 'bot',
        text: response.response || 'I couldn\'t generate a response. Please try again.',
        sources: response.sources || [],
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        type: 'bot',
        text: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
      console.error('Query error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const suggestedQuestions = [
    'What are the key findings in my report?',
    'Are there any abnormal values?',
    'What do these results mean?',
    'What should I discuss with my doctor?'
  ];

  const handleSuggestedQuestion = (question) => {
    setInputText(question);
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h2>💬 Medical Report Assistant</h2>
        <p>Ask questions about your medical report</p>
      </div>

      {messages.length === 1 && (
        <div className="suggested-questions">
          <p><strong>Suggested questions:</strong></p>
          <div className="question-buttons">
            {suggestedQuestions.map((question, index) => (
              <button
                key={index}
                className="question-button"
                onClick={() => handleSuggestedQuestion(question)}
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="messages-container">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.type}`}>
            <div className="message-content">
              <p>{message.text}</p>
              {message.sources && message.sources.length > 0 && (
                <div className="message-sources">
                  <small>📚 Sources: {message.sources.join(', ')}</small>
                </div>
              )}
            </div>
            <div className="message-timestamp">
              {message.timestamp.toLocaleTimeString()}
            </div>
          </div>
        ))}
        {loading && (
          <div className="message bot">
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-container">
        <textarea
          className="chat-input"
          placeholder="Ask a question about your report..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={loading}
          rows="2"
        />
        <button
          className="send-button"
          onClick={handleSendMessage}
          disabled={!inputText.trim() || loading}
        >
          {loading ? '⏳' : '📤'} Send
        </button>
      </div>

      <div className="chat-disclaimer">
        <small>
          ⚠️ This is an AI assistant for educational purposes only. 
          Always consult with healthcare professionals for medical advice.
        </small>
      </div>
    </div>
  );
}

export default ChatInterface;
