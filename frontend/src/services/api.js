/**
 * API service for Medical Report AI frontend
 * Connects to the backend Flask API
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

/**
 * Upload a medical report file
 * @param {File} file - The file to upload
 * @returns {Promise<Object>} Response with file_id and extracted_text
 */
export async function uploadReport(file) {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${API_BASE_URL}/api/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Upload failed');
    }

    return await response.json();
  } catch (error) {
    console.error('Upload error:', error);
    throw error;
  }
}

/**
 * Query the uploaded report with a question
 * @param {string} query - The question to ask
 * @param {string} fileId - The uploaded file ID
 * @returns {Promise<Object>} Response with answer and sources
 */
export async function queryReport(query, fileId = '') {
  console.log('🔍 Query API - FileID:', fileId, 'Query:', query);
  try {
    const response = await fetch(`${API_BASE_URL}/api/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        file_id: fileId,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Query failed');
    }

    return await response.json();
  } catch (error) {
    console.error('Query error:', error);
    throw error;
  }
}

/**
 * Analyze medical report text
 * @param {string} text - The report text to analyze
 * @returns {Promise<Object>} Analysis results
 */
export async function analyzeReport(text) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Analysis failed');
    }

    return await response.json();
  } catch (error) {
    console.error('Analysis error:', error);
    throw error;
  }
}

/**
 * Check API health status
 * @returns {Promise<Object>} Health status
 */
export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    return await response.json();
  } catch (error) {
    console.error('Health check error:', error);
    return { status: 'offline' };
  }
}

export default {
  uploadReport,
  queryReport,
  analyzeReport,
  checkHealth,
};
