import React, { useState } from 'react';
import { uploadReport } from '../services/api';
import './UploadReport.css';

function UploadReport({ onUploadSuccess }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [preview, setPreview] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    
    if (!file) return;

    // Validate file type
    const allowedTypes = ['application/pdf', 'image/png', 'image/jpeg', 'image/jpg'];
    if (!allowedTypes.includes(file.type)) {
      setError('Please upload a PDF or image file (PNG, JPG, JPEG)');
      return;
    }

    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
      setError('File size must be less than 16MB');
      return;
    }

    setSelectedFile(file);
    setError('');

    // Create preview for images
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    } else {
      setPreview(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first');
      return;
    }

    setUploading(true);
    setError('');

    try {
      const response = await uploadReport(selectedFile);
      
      if (response.success) {
        onUploadSuccess(response.file_id, response.extracted_text);
        setSelectedFile(null);
        setPreview(null);
        
        // Show blood analysis if available
        if (response.blood_analysis && response.blood_analysis.success) {
          const analysis = response.blood_analysis;
          let message = `✅ Report analyzed!\n\nTests found: ${analysis.total_tests}\n`;
          
          if (analysis.abnormal_count > 0) {
            message += `⚠️ Abnormal results: ${analysis.abnormal_count}\n\n`;
            message += 'Switch to Chat tab to ask questions about your results!';
          } else {
            message += '✅ All values within normal ranges!';
          }
          
          alert(message);
        }
      } else {
        setError(response.error || 'Upload failed');
      }
    } catch (err) {
      setError('Failed to upload file. Please try again.');
      console.error('Upload error:', err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-container">
      <div className="upload-box">
        <h2>Upload Medical Report</h2>
        <p className="upload-description">
          Upload your medical report (PDF, PNG, JPG) to extract information and ask questions
        </p>

        <div className="file-input-wrapper">
          <input
            type="file"
            id="file-input"
            accept=".pdf,.png,.jpg,.jpeg"
            onChange={handleFileSelect}
            disabled={uploading}
          />
          <label htmlFor="file-input" className="file-input-label">
            📁 Choose File
          </label>
        </div>

        {selectedFile && (
          <div className="file-info">
            <p><strong>Selected file:</strong> {selectedFile.name}</p>
            <p><strong>Size:</strong> {(selectedFile.size / 1024).toFixed(2)} KB</p>
          </div>
        )}

        {preview && (
          <div className="image-preview">
            <img src={preview} alt="Preview" />
          </div>
        )}

        {error && (
          <div className="error-message">
            ⚠️ {error}
          </div>
        )}

        <button
          className="upload-button"
          onClick={handleUpload}
          disabled={!selectedFile || uploading}
        >
          {uploading ? '⏳ Uploading...' : '🚀 Upload & Process'}
        </button>

        <div className="upload-info">
          <h3>Supported Formats:</h3>
          <ul>
            <li>PDF documents</li>
            <li>Images (PNG, JPG, JPEG)</li>
          </ul>
          <p><strong>Maximum file size:</strong> 16MB</p>
        </div>
      </div>
    </div>
  );
}

export default UploadReport;
