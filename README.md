# algorithms.github.io

Visual summaries of data structures, algorithms, and patterns — built for intuition. This repository powers a GitHub Pages site dedicated to clear, visual, and intuitive explanations of core CS ideas.

## 🚀 Live Site

This site is automatically deployed to GitHub Pages at: **https://harihari1989.github.io/**

## 🛠️ Development

This is a React application built with Vite and TypeScript.

### Local Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 📦 Deployment

The site automatically deploys to GitHub Pages via GitHub Actions when changes are pushed to the `main` branch. The workflow:

1. Builds the React app using Vite
2. Uploads the `dist/` folder as a Pages artifact
3. Deploys to GitHub Pages

See `.github/workflows/deploy.yml` for the full workflow configuration.

## 🏗️ Tech Stack

- **React** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **React Router** - Client-side routing
- **GitHub Pages** - Hosting platform
