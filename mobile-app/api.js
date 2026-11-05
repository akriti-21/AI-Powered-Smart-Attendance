import axios from 'axios';
import { BASE_URL } from './config';

console.log('API BASE URL:', BASE_URL);

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60000,
});

export function getApiErrorMessage(error, fallback = 'Something went wrong') {
  console.log('API ERROR:', error);
  console.log('API ERROR MESSAGE:', error?.message);
  console.log('API ERROR CODE:', error?.code);
  console.log('API ERROR RESPONSE:', error?.response?.data);
  console.log('API ERROR STATUS:', error?.response?.status);
  console.log('API BASE URL USED:', api.defaults.baseURL);

  if (error.response) {
    const detail = error.response.data?.detail;

    if (typeof detail === 'string') {
      return detail;
    }

    if (Array.isArray(detail)) {
      return detail
        .map((d) => `${(d.loc || []).join('.')}: ${d.msg}`)
        .join('\n');
    }

    if (detail) {
      return JSON.stringify(detail);
    }

    return `Request failed with status ${error.response.status}`;
  }

  if (error.request) {
    return `Cannot reach ${api.defaults.baseURL}/students/`;
  }

  return error.message || fallback;
}

export default api;