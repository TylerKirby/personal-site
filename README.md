# Deploy blog
From this dir, run 
```bash
npm run build
cp -a public/* ~/Projects/tylerkirby.github.io
cd ~/Projects/tylerkirby.github.io
git add .
git commit -m "blog update"
git push
```