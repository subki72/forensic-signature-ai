/**
 * API Service Client for Forensic Signature AI.
 * Handles network requests, timeout aborts, and standardized error parsing.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const DEFAULT_TIMEOUT_MS = 20000;

/**
 * Perform a fetch with configurable timeout using AbortController.
 */
async function fetchWithTimeout(url, options = {}, timeoutMs = DEFAULT_TIMEOUT_MS) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return response;
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new Error(
        `Request timed out after ${timeoutMs / 1000} seconds. The backend may be processing heavy tasks.`,
        { cause: err }
      );
    }
    throw err;
  } finally {
    clearTimeout(id);
  }
}

/**
 * Check backend liveness probe.
 */
export async function checkLiveness() {
  try {
    const res = await fetchWithTimeout(`${API_BASE_URL}/health`, { method: 'GET' }, 5000);
    if (!res.ok) return { online: false, status: 'unhealthy' };
    const data = await res.json();
    return { online: true, data };
  } catch (err) {
    return { online: false, error: err.message };
  }
}

/**
 * Check backend model readiness probe.
 */
export async function checkReadiness() {
  try {
    const res = await fetchWithTimeout(`${API_BASE_URL}/ready`, { method: 'GET' }, 5000);
    if (!res.ok) return { ready: false, status: 'degraded' };
    const data = await res.json();
    return { ready: data.model_loaded, data };
  } catch (err) {
    return { ready: false, error: err.message };
  }
}

/**
 * Verify two signature specimens.
 */
export async function verifySignatures(referenceFile, questionedFile) {
  if (!referenceFile || !questionedFile) {
    throw new Error('Both reference and questioned signature documents must be provided.');
  }

  // Client-side file size pre-check (5MB)
  const maxBytes = 5 * 1024 * 1024;
  if (referenceFile.size > maxBytes) {
    throw new Error(`Reference document exceeds 5 MB size limit (${(referenceFile.size / (1024 * 1024)).toFixed(1)} MB).`);
  }
  if (questionedFile.size > maxBytes) {
    throw new Error(`Questioned document exceeds 5 MB size limit (${(questionedFile.size / (1024 * 1024)).toFixed(1)} MB).`);
  }

  const formData = new FormData();
  formData.append('file_asli', referenceFile);
  formData.append('file_uji', questionedFile);

  const res = await fetchWithTimeout(`${API_BASE_URL}/verify`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    let errorDetail = `HTTP ${res.status}: ${res.statusText}`;
    try {
      const errorJson = await res.json();
      if (errorJson.error) {
        errorDetail = errorJson.error;
      } else if (errorJson.detail) {
        errorDetail = Array.isArray(errorJson.detail)
          ? errorJson.detail.map((d) => d.msg || JSON.stringify(d)).join(', ')
          : errorJson.detail;
      }
    } catch {
      // Use fallback errorDetail
    }
    throw new Error(errorDetail);
  }

  const data = await res.json();
  return data.verification;
}
