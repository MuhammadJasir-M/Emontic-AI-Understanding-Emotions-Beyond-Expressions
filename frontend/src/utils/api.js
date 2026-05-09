// src/utils/api.js
// Axios instance configured for the Emontic AI backend.

import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_URL,
  timeout: 30000, // 30s — allows for cold-start model loading
  headers: {
    Accept: 'application/json',
  },
});

export default api;
export { API_URL };
