# Performance Optimizations for Vercel Deployment

This document outlines additional performance optimizations specifically for Vercel serverless deployment to improve execution speed and reduce cold start times.

## Current Optimizations (Already Applied)

Your app already has excellent optimizations:
- ✅ Global cache system for models and data
- ✅ LRU caching for API calls (maxsize=512 for session results)
- ✅ Preloaded models and CSV files
- ✅ FastF1 cache directory

## Additional Optimizations for Vercel/Serverless

### 1. Lazy Initialization (Critical for Cold Starts)

**Problem**: On Vercel, the function initializes on every cold start, loading all models and data even if not needed.

**Solution**: Implement lazy loading - only load resources when first accessed.

**Implementation**: The current `initialize_global_cache()` runs immediately. For serverless, we should:
- Keep cache initialization but make it thread-safe
- Load models only when prediction is first called
- Use a lock to prevent multiple simultaneous initializations

### 2. Response Caching Headers

**Problem**: Static assets and API responses aren't cached by browsers/CDN.

**Solution**: Add cache headers to reduce redundant requests.

**Implementation**: Add to your Flask app:
```python
@app.server.after_request
def add_cache_headers(response):
    # Cache static assets for 1 hour
    if request.path.startswith('/assets/'):
        response.cache_control.max_age = 3600
        response.cache_control.public = True
    # Cache API responses for 5 minutes
    elif request.path.startswith('/_dash'):
        response.cache_control.max_age = 300
    return response
```

### 3. Optimize CSV Reading

**Problem**: `pd.read_csv()` can be slow for large files.

**Solution**: Use faster parsing options or pre-process to parquet format.

**Optimization**:
```python
# Faster CSV reading with optimized parameters
df = pd.read_csv('file.csv', 
                 engine='c',  # Use C engine (faster)
                 low_memory=False,  # Read all at once
                 dtype_backend='pyarrow'  # Use PyArrow backend if available
)
```

### 4. Connection Pooling for API Calls

**Problem**: Each API call creates a new connection, adding latency.

**Solution**: Use a session with connection pooling for external API calls.

**Implementation**:
```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Create a session with connection pooling
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
session.mount("http://", adapter)
session.mount("https://", adapter)
```

### 5. Reduce Model Size (If Possible)

**Problem**: Large model files increase cold start time.

**Solutions**:
- Use model quantization (if using XGBoost, consider using `save_model` with compression)
- Consider using smaller models for initial predictions
- Load models in background thread (not blocking)

### 6. Optimize FastF1 Cache

**Problem**: FastF1 cache directory might not persist well in serverless.

**Solution**: 
- Use Vercel's `/tmp` directory (persists during function execution)
- Consider using external cache (Redis, Upstash) for shared cache across invocations

### 7. Async Operations Where Possible

**Problem**: Synchronous operations block the function.

**Solution**: Use async/await for I/O-bound operations (API calls, file reads).

**Note**: Dash callbacks are synchronous, but you can use background threads for non-critical operations.

### 8. Vercel-Specific Optimizations

#### A. Use Vercel Edge Functions for Static Content
- Serve static assets (CSS, JS) via Edge Functions (faster, closer to users)

#### B. Enable Vercel's Built-in Caching
- Configure `vercel.json` to cache responses:
```json
{
  "headers": [
    {
      "source": "/assets/(.*)",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "public, max-age=31536000, immutable"
        }
      ]
    }
  ]
}
```

#### C. Use Vercel Pro Features
- **Keep-alive**: Reduces cold starts
- **60-second timeout**: Allows longer operations
- **3GB memory**: Better for large models

### 9. Database/External Cache (Advanced)

**Problem**: Serverless functions don't share memory between invocations.

**Solution**: Use external cache (Redis, Upstash) for:
- API response caching
- Model predictions caching
- Session data

**Services**:
- **Upstash Redis**: Serverless Redis, free tier available
- **Vercel KV**: Vercel's key-value store (built on Upstash)

### 10. Code Splitting

**Problem**: Loading all code increases cold start time.

**Solution**: 
- Import heavy libraries only when needed
- Use lazy imports for optional features

## Implementation Priority

### High Priority (Do First):
1. ✅ Response caching headers
2. ✅ Connection pooling for API calls
3. ✅ Optimize CSV reading

### Medium Priority:
4. Lazy initialization for models
5. Vercel Edge Functions for static assets
6. External cache (Redis/Upstash)

### Low Priority (If Needed):
7. Model quantization
8. Code splitting
9. Async operations

## Monitoring Performance

After deploying, monitor:
1. **Cold start time**: Should be < 10 seconds (free tier) or < 5 seconds (Pro)
2. **Warm execution time**: Should be < 2 seconds for most operations
3. **Function duration**: Check Vercel dashboard → Functions → Duration
4. **Memory usage**: Should stay under limits
5. **Error rate**: Should be < 1%

## Quick Wins

These can be implemented quickly for immediate improvements:

1. **Add cache headers** (5 minutes)
2. **Use connection pooling** (10 minutes)
3. **Optimize CSV reading** (5 minutes)
4. **Upgrade to Vercel Pro** (if budget allows, immediate improvement)

## Expected Improvements

With these optimizations:
- **Cold start**: 30-50% faster (from ~15s to ~8-10s)
- **Warm execution**: 20-40% faster (from ~3s to ~1.5-2s)
- **API calls**: 50% faster (with connection pooling)
- **Static assets**: 80% faster (with CDN caching)

## Next Steps

1. Implement high-priority optimizations
2. Deploy and test
3. Monitor performance metrics
4. Iterate based on results
