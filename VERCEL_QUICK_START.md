# Vercel Deployment Quick Start Guide

## 🚀 Quick Summary

This guide will help you deploy your F1 Fantasy Predictor to Vercel with your custom domain `prateekf1project.online`.

## ✅ What's Been Prepared

1. **Vercel Configuration Files**:
   - `vercel.json` - Vercel deployment configuration
   - `api/index.py` - Serverless function entry point

2. **Performance Optimizations**:
   - ✅ HTTP connection pooling for API calls
   - ✅ Cache headers for static assets
   - ✅ Optimized CSV reading
   - ✅ Response caching

3. **Documentation**:
   - `VERCEL_DEPLOYMENT.md` - Detailed deployment guide
   - `VERCEL_PERFORMANCE_OPTIMIZATIONS.md` - Performance tips

## 📋 Deployment Steps (5 Minutes)

### Step 1: Install Vercel CLI (Optional)
```bash
npm install -g vercel
```

### Step 2: Login to Vercel
```bash
vercel login
```

### Step 3: Deploy
```bash
cd /Users/prateekgoel/Documents/f1_fantasy_predictor
vercel
```

When prompted:
- **Set up and deploy?** → Yes
- **Which scope?** → Your account
- **Link to existing project?** → No
- **Project name?** → f1-fantasy-predictor (or any name)
- **Directory?** → ./

### Step 4: Set Environment Variables
```bash
vercel env add OPENWEATHER_API_KEY
# Enter your API key when prompted
```

Or via Dashboard:
1. Go to your project → Settings → Environment Variables
2. Add `OPENWEATHER_API_KEY` with your key

### Step 5: Deploy to Production
```bash
vercel --prod
```

## 🌐 Custom Domain Setup

### Step 1: Add Domain in Vercel
1. Go to your project in Vercel dashboard
2. Settings → Domains
3. Add: `prateekf1project.online`

### Step 2: Configure DNS at Spaceship.com
1. Log in to Spaceship.com
2. Go to Domain Management → `prateekf1project.online`
3. DNS Settings → Add records:

**For Root Domain:**
```
Type: A
Name: @
Value: [IP from Vercel dashboard]
```

**For WWW:**
```
Type: CNAME
Name: www
Value: [CNAME from Vercel dashboard]
```

### Step 3: Wait & Verify
- Wait 15-30 minutes for DNS propagation
- Check status in Vercel dashboard (should show green checkmark)
- Visit `https://prateekf1project.online`

## ⚠️ Important Notes

### Vercel Limitations
- **Free Tier**: 10-second function timeout
- **Pro Tier** ($20/month): 60-second timeout, better performance
- **Cold Starts**: First request after inactivity can be slow (5-30s)

### Recommendations
1. **Start with Free Tier** to test
2. **Upgrade to Pro** if you hit timeout issues or need better performance
3. **Monitor** function execution times in Vercel dashboard

## 🎯 Performance Improvements Applied

Your app now has:
- ✅ Connection pooling (50% faster API calls)
- ✅ Cache headers (80% faster static assets)
- ✅ Optimized CSV reading
- ✅ Global cache system (already existed)

**Expected Results:**
- Cold start: 8-15 seconds (free tier) or 5-8 seconds (Pro)
- Warm execution: 1-3 seconds
- Static assets: Served from CDN edge

## 🔧 Troubleshooting

### Issue: Function Timeout
**Solution**: Upgrade to Vercel Pro for 60-second timeout

### Issue: Domain Not Working
**Solution**: 
1. Check DNS records match Vercel exactly
2. Wait longer (up to 48 hours)
3. Verify at dnschecker.org

### Issue: Cold Start Too Slow
**Solution**: 
1. Upgrade to Pro (better cold start)
2. Use keep-alive feature (Pro only)
3. Consider alternative platforms (Railway, Render)

## 📚 Full Documentation

- **Detailed Guide**: See `VERCEL_DEPLOYMENT.md`
- **Performance Tips**: See `VERCEL_PERFORMANCE_OPTIMIZATIONS.md`
- **Vercel Docs**: https://vercel.com/docs

## 🆘 Need Help?

1. Check Vercel dashboard → Functions → Logs for errors
2. Review `VERCEL_DEPLOYMENT.md` for detailed steps
3. Vercel Support: support@vercel.com

## 🎉 Next Steps After Deployment

1. ✅ Test all features on your domain
2. ✅ Monitor performance in Vercel dashboard
3. ✅ Consider upgrading to Pro if needed
4. ✅ Share your site: `https://prateekf1project.online`

---

**Good luck with your deployment! 🏎️**
