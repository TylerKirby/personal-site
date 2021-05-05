# Deploy blog
Build to `docs` and run ` gh-pages -t -b master -d docs`.

From this dir, run 
```bash
npm run build
gh-pages -t -b master -d docs
cp -a docs/* ~/Projects/tylerkirby.github.io
cd ~/Projects/tylerkirby.github.io
git add .
git commit -m "blog update"
git push
```