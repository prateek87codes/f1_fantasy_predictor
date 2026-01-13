# Performance Optimizations Applied

This document outlines all the performance optimizations made to improve app loading times and prevent pages from failing to load.

## Summary of Changes

### 1. Global Cache System (CRITICAL)
- **Problem**: Models and CSV files were loaded on-demand in callbacks, causing repeated disk I/O operations
- **Solution**: Created `GLOBAL_CACHE` dictionary that preloads all expensive resources at app startup
- **Impact**: Eliminates redundant file reads - models and data loaded once at startup, reused across all requests

**What's cached:**
- `f1_prediction_model_enhanced.joblib` - Enhanced ML model
- `f1_prediction_features_enhanced.joblib` - Feature metadata
- `f1_prediction_model.joblib` - Basic ML model (fallback)
- `f1_historical_data.csv` - Historical race data
- `circuit_data.csv` - Circuit information

### 2. FastF1 Cache Directory Fix
- **Problem**: Cache was using temporary directory which doesn't persist properly in GCP
- **Solution**: Changed to persistent `cache/` directory
- **Impact**: FastF1 API responses are cached persistently, reducing API calls

### 3. Increased LRU Cache Sizes
- **Problem**: Small cache sizes (maxsize=2, maxsize=32) caused frequent cache misses
- **Solution**: Increased cache sizes significantly:
  - `get_session_results`: maxsize=2 → maxsize=512 (256x increase)
  - `get_all_teams_data`: maxsize=2 → maxsize=128 (64x increase)
  - `get_next_race_info`: maxsize=32 → maxsize=128 (4x increase)
  - `get_championship_standings_progression`: maxsize=32 → maxsize=128 (4x increase)
- **Impact**: More requests served from cache, reducing API calls and database queries

### 4. Replaced On-Demand Model Loading
- **Problem**: Every prediction callback loaded models from disk
- **Solution**: All callbacks now use cached models from `GLOBAL_CACHE`
- **Impact**: Prediction operations are now instant after initial load

### 5. Replaced Repeated CSV Reads
- **Problem**: CSV files read multiple times per request
- **Solution**: All callbacks use cached DataFrames from `GLOBAL_CACHE`
- **Impact**: Eliminates redundant file I/O operations

### 6. Gunicorn Configuration Optimization
- **Changes**:
  - Increased workers: 2 → 4 (better concurrency)
  - Increased timeout: 120s → 180s (prevents timeout on slow API calls)
  - Added `preload_app = True` (critical for cache sharing)
  - Added worker recycling (`max_requests = 1000`) to prevent memory leaks
- **Impact**: Better handling of concurrent requests and slower operations

### 7. App Engine Configuration Optimization
- **Changes**:
  - Increased memory: 2GB → 4GB (better caching and model loading)
  - Increased health check timeouts: 4s → 10s (prevents false positives)
  - Adjusted CPU utilization targets: 0.6 → 0.7
- **Impact**: More resources available for caching, less likely to crash on slow loads

## Performance Improvements Expected

1. **Initial Page Load**: 50-70% faster (models and data preloaded)
2. **Subsequent Requests**: 80-90% faster (served from cache)
3. **Prediction Operations**: 90%+ faster (no disk I/O)
4. **Page Reliability**: Should eliminate timeout-related failures

## Monitoring

After deployment, monitor:
- Memory usage (should stay under 4GB)
- Response times (should decrease significantly)
- Cache hit rates (check logs for cache usage messages)
- Error rates (should decrease)

## Rollback Plan

If issues occur:
1. Revert `app.yaml` to original configuration
2. Reduce workers in `gunicorn_config.py` back to 2
3. Reduce memory back to 2GB if needed

## Notes

- The `preload_app = True` setting in Gunicorn is critical - it ensures the cache is initialized before workers are forked
- Cache sizes can be adjusted based on actual usage patterns
- Consider implementing cache warming on startup if needed

## Recent Updates (2025-10-30)

- Fantasy Team Creator images
  - Fixed driver image filenames to use lowercase (e.g., `Nor.png` → `nor.png`).
  - Result: Images render correctly across pages.

- Predictions tab performance
  - Added seasonal simulation result caching in `GLOBAL_CACHE['simulation_results']` so the RL simulation runs once and reuses results.
  - Kept synchronous rendering of the simulation UI; first load may take ~30–60s; subsequent loads are instant.

- AI highlights (Perplexity)
  - Restored original synchronous API usage so highlights appear when responses return.
  - Added resilient calling with 3-attempt retry and exponential backoff; timeout increased to 15s.
  - Root cause of current AI failures confirmed via Cloud Run logs: `401 Unauthorized` from Perplexity due to exhausted account credits.
  - Current state: Code and deployment are ready; once credits are topped up (or a fresh key is provided in Secret Manager), AI summaries will render again without code changes.

- Deployment
  - Rebuilt and redeployed to Cloud Run multiple times to propagate fixes.
  - Gunicorn still uses `preload_app = True`; workers=4; timeout=180s.

## Next Steps (post-credit top-up)

1. Update or rotate the `PERPLEXITY_API_KEY` secret with a funded key.
2. Redeploy (or simply trigger a new revision) so the service picks up the secret.
3. Verify AI highlights on:
   - Current Season → Season So Far
   - Past Seasons (race summaries)
   - Teams & Drivers (History/Performance)
   - Predictions → Next Race Tyre Strategy

No additional code changes are required after credits are added.
