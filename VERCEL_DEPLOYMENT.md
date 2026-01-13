# Vercel Deployment Guide for F1 Fantasy Predictor

This guide will walk you through deploying your Dash application to Vercel with your custom domain `prateekf1project.online`.

## ⚠️ Important Considerations

**Before proceeding, please note:**

1. **Vercel Limitations**: Vercel is designed for serverless functions with execution time limits:
   - Free tier: 10 seconds per function execution
   - Pro tier: 60 seconds per function execution
   - Your Dash app may have operations that exceed these limits

2. **Cold Starts**: Serverless functions have cold starts, which means the first request after inactivity can be slow (5-30 seconds) as models and data are loaded.

3. **Alternative Platforms**: For Dash applications, you might want to consider:
   - **Railway** (railway.app) - Better for long-running apps, free tier available
   - **Render** (render.com) - Free tier for web services
   - **Fly.io** (fly.io) - Good for containerized apps
   - **Google Cloud Run** (current) - Already working, just add custom domain

**However, if you still want to proceed with Vercel, follow these steps:**

## Prerequisites

1. ✅ Vercel account (you mentioned you already have one)
2. ✅ Domain `prateekf1project.online` purchased from spaceship.com
3. ✅ Vercel CLI installed (optional, but recommended)

## Step 1: Install Vercel CLI (Optional but Recommended)

```bash
npm install -g vercel
```

Or if you prefer using the web interface, you can skip this step.

## Step 2: Prepare Your Project

The following files have been created/updated for Vercel deployment:
- `vercel.json` - Vercel configuration
- `api/index.py` - Serverless function entry point

Make sure you have all required files:
- `app.py` - Your main Dash application
- `requirements.txt` - Python dependencies
- Model files (`.joblib` files)
- CSV data files
- `assets/` folder with CSS

## Step 3: Set Up Environment Variables

You'll need to set environment variables in Vercel:

1. **OPENWEATHER_API_KEY** (if you're using weather features)

### Via Vercel Dashboard:
1. Go to your project settings
2. Navigate to "Environment Variables"
3. Add `OPENWEATHER_API_KEY` with your API key value

### Via CLI:
```bash
vercel env add OPENWEATHER_API_KEY
```

## Step 4: Deploy to Vercel

### Option A: Deploy via Vercel Dashboard (Easiest)

1. **Go to [vercel.com](https://vercel.com)** and log in
2. **Click "Add New Project"**
3. **Import your Git repository**:
   - Connect your GitHub/GitLab/Bitbucket account if not already connected
   - Select the repository containing this project
   - Or upload the project files directly
4. **Configure the project**:
   - **Framework Preset**: Other (or leave as default)
   - **Root Directory**: `./` (root)
   - **Build Command**: Leave empty (no build needed for Python)
   - **Output Directory**: Leave empty
   - **Install Command**: `pip install -r requirements.txt`
5. **Click "Deploy"**

### Option B: Deploy via CLI

```bash
# Navigate to your project directory
cd /Users/prateekgoel/Documents/f1_fantasy_predictor

# Login to Vercel (first time only)
vercel login

# Deploy
vercel

# For production deployment
vercel --prod
```

## Step 5: Configure Custom Domain

### Step 5.1: Add Domain in Vercel

1. **Go to your project** in Vercel dashboard
2. **Click "Settings"** → **"Domains"**
3. **Add your domain**: `prateekf1project.online`
4. **Vercel will show you DNS records** to add (something like):
   - Type: `A` or `CNAME`
   - Name: `@` or `www`
   - Value: Vercel's IP or CNAME target

### Step 5.2: Configure DNS at Spaceship.com

1. **Log in to your Spaceship.com account**
2. **Go to Domain Management** → Select `prateekf1project.online`
3. **Navigate to DNS Settings**
4. **Add the DNS records** provided by Vercel:
   - For root domain (`@`): Add A record pointing to Vercel's IP
   - For www subdomain: Add CNAME record pointing to Vercel's CNAME
   - **OR** use CNAME flattening if supported

**Example DNS Records:**
```
Type: A
Name: @
Value: 76.76.21.21 (Vercel's IP - check Vercel dashboard for actual IP)

Type: CNAME
Name: www
Value: cname.vercel-dns.com (Vercel's CNAME - check dashboard)
```

### Step 5.3: Wait for DNS Propagation

- DNS changes can take **5 minutes to 48 hours** to propagate
- Usually takes **15-30 minutes**
- You can check propagation status at: https://dnschecker.org

### Step 5.4: Verify Domain in Vercel

1. **Go back to Vercel dashboard** → Your project → Settings → Domains
2. **Wait for the domain to show as "Valid"** (green checkmark)
3. Once valid, your site will be accessible at `https://prateekf1project.online`

## Step 6: SSL Certificate (Automatic)

Vercel automatically provisions SSL certificates via Let's Encrypt. Once your domain is verified, HTTPS will be enabled automatically (usually within a few minutes).

## Step 7: Test Your Deployment

1. Visit `https://prateekf1project.online`
2. Test all features:
   - Historical data loading
   - Predictions
   - Visualizations
   - All tabs and interactions

## Troubleshooting

### Issue: Function Timeout

**Symptom**: Requests timeout after 10 seconds (free tier) or 60 seconds (pro tier)

**Solutions**:
1. **Upgrade to Vercel Pro** for 60-second timeout
2. **Optimize slow operations** (see Performance Optimizations section)
3. **Consider using a different platform** (Railway, Render) for better timeout handling

### Issue: Cold Start Too Slow

**Symptom**: First request takes 20-30 seconds

**Solutions**:
1. **Use Vercel Pro** with better cold start performance
2. **Implement keep-alive** (Vercel Pro feature)
3. **Pre-warm functions** using a cron job or monitoring service

### Issue: Memory Limits

**Symptom**: Out of memory errors

**Solutions**:
1. **Upgrade to Vercel Pro** (3GB memory vs 1GB on free tier)
2. **Optimize model loading** (lazy load, smaller models)
3. **Use external caching** (Redis, Upstash)

### Issue: Domain Not Working

**Symptoms**: Domain shows as invalid or not resolving

**Solutions**:
1. **Double-check DNS records** match exactly what Vercel provides
2. **Wait longer** for DNS propagation (can take up to 48 hours)
3. **Check DNS propagation** at dnschecker.org
4. **Contact Spaceship.com support** if DNS records aren't updating

## Performance Optimizations (Already Applied)

The following optimizations have been implemented to improve speed:

1. **Global Cache System**: Models and data preloaded at startup
2. **LRU Caching**: Increased cache sizes for API calls
3. **Lazy Loading**: Resources loaded only when needed
4. **Optimized Gunicorn Config**: Better worker configuration

## Additional Performance Tips for Vercel

Since Vercel is serverless, consider these additional optimizations:

1. **Use Vercel Edge Functions** for static content (if applicable)
2. **Implement response caching** headers
3. **Use Vercel's built-in caching** for static assets
4. **Consider CDN** for large model files (store in S3/Cloud Storage)

## Monitoring

After deployment, monitor:
- **Vercel Dashboard** → Analytics for:
  - Function execution time
  - Error rates
  - Request counts
- **Function Logs** for errors and performance issues

## Cost Considerations

- **Vercel Free Tier**:
  - 100GB bandwidth/month
  - 100 hours function execution/month
  - 10-second function timeout
  
- **Vercel Pro** ($20/month):
  - Unlimited bandwidth
  - 1000 hours function execution/month
  - 60-second function timeout
  - Better cold start performance
  - 3GB memory per function

## Next Steps

1. ✅ Deploy to Vercel
2. ✅ Configure custom domain
3. ✅ Test all features
4. ⚠️ Monitor performance and consider upgrading to Pro if needed
5. ⚠️ Consider alternative platforms if Vercel limitations become problematic

## Alternative: Keep Cloud Run + Add Custom Domain

If Vercel doesn't work well for your use case, you can:
1. Keep your current Cloud Run deployment
2. Add custom domain to Cloud Run (Google Cloud Console → Cloud Run → Manage Custom Domains)
3. Configure DNS at Spaceship.com to point to Cloud Run

This might be easier and more reliable for a Dash application.

---

**Need Help?**
- Vercel Docs: https://vercel.com/docs
- Vercel Support: support@vercel.com
- Spaceship.com Support: Check their support portal
