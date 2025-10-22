# 🎨 LLM Fine-Tuning Pipeline - Frontend

Modern, responsive React/Next.js frontend with real-time training monitoring and beautiful UI.

## ✨ Features

- **🎯 6-Step Wizard**: Intuitive pipeline interface
- **🔄 Real-Time Updates**: WebSocket-powered training monitoring
- **📊 Interactive Charts**: Live loss curves and metrics visualization
- **🎨 Beautiful UI**: Glassmorphism design with smooth animations
- **📱 Responsive**: Works on desktop, tablet, and mobile
- **⚡ Fast**: Optimized with Next.js 14 App Router
- **🎭 Type-Safe**: Full TypeScript coverage
- **♿ Accessible**: ARIA labels and keyboard navigation

## 🏗️ Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: Radix UI (shadcn/ui)
- **State Management**: Zustand
- **Data Fetching**: TanStack Query (React Query)
- **API Client**: Axios
- **WebSocket**: Socket.IO Client
- **Charts**: Recharts
- **Icons**: Lucide React
- **Animations**: Framer Motion

## 📋 Prerequisites

- Node.js 18 or higher
- npm or yarn
- Backend API running on http://localhost:8000

## 🛠️ Installation

### 1. Install Dependencies

```bash
cd frontend
npm install
```

Or with yarn:

```bash
yarn install
```

### 2. Configure Environment

```bash
# Copy example env file
copy .env.local.example .env.local

# Edit .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

## 🚀 Running the Application

### Development Mode

```bash
npm run dev
```

The application will be available at:
- **Frontend**: http://localhost:3000
- **Landing Page**: http://localhost:3000
- **Pipeline**: http://localhost:3000/pipeline

### Production Build

```bash
# Build for production
npm run build

# Start production server
npm run start
```

### Type Checking

```bash
npm run type-check
```

### Linting

```bash
npm run lint
```

## 📱 Application Structure

```
frontend/
├── app/
│   ├── layout.tsx              # Root layout
│   ├── page.tsx                # Landing page
│   ├── providers.tsx           # React Query provider
│   ├── globals.css             # Global styles
│   └── pipeline/
│       └── page.tsx            # Main pipeline interface
├── components/
│   ├── ui/                     # Reusable UI components
│   ├── ModelSelector.tsx       # Model selection component
│   ├── DatasetUploader.tsx     # Dataset upload component
│   ├── HyperparameterConfig.tsx
│   ├── TrainingMonitor.tsx     # Real-time training display
│   ├── QuantizationOptions.tsx
│   └── ExportPanel.tsx
├── lib/
│   ├── api.ts                  # API client
│   ├── websocket.ts            # WebSocket manager
│   └── utils.ts                # Utility functions
├── hooks/
│   ├── useTraining.ts          # Training hook
│   ├── useModelAnalysis.ts     # Model analysis hook
│   └── useWebSocket.ts         # WebSocket hook
├── store/
│   └── pipelineStore.ts        # Zustand state store
└── types/
    └── index.ts                # TypeScript types
```

## 🎨 UI Components

### Landing Page

- **Hero Section**: Animated gradient background, CTA buttons
- **Features Grid**: 6 key features with icons
- **How It Works**: Step-by-step process
- **Stats**: Key metrics display
- **CTA Section**: Final call-to-action

### Pipeline Interface

**Step 1: Model Selection**
- Search autocomplete for HF models
- Popular models grid
- Model info cards
- "Analyze Model" button
- Results display

**Step 2: Dataset Upload**
- Drag-and-drop zone
- File format selector
- HF dataset ID input
- Validation results
- Preview table

**Step 3: Hyperparameter Configuration**
- "Get Recommendations" button
- AI-recommended config display
- Interactive parameter sliders
- Memory requirement visualization
- Training time estimator

**Step 4: Training Monitor**
- Real-time loss curves
- Live metrics dashboard
- GPU utilization chart
- Training logs terminal
- Progress indicators
- Checkpoint list

**Step 5: Quantization**
- Method selector (4-bit, 8-bit, GPTQ, GGUF)
- Comparison table
- Size reduction visualization
- "Quantize Model" button

**Step 6: Export & Deploy**
- Training summary
- Generated code viewers
- Download buttons
- Copy-to-clipboard
- Package download

## 🔌 API Integration

### API Client (`lib/api.ts`)

```typescript
import { apiClient } from '@/lib/api';

// Analyze model
const modelInfo = await apiClient.analyzeModel('gpt2');

// Upload dataset
const datasetInfo = await apiClient.uploadDataset(file, 'json');

// Get recommendations
const recommendations = await apiClient.recommendHyperparameters(
  'gpt2',
  'my_dataset',
  'free',
  'text-generation'
);

// Start training
const job = await apiClient.startTraining(config);
```

### WebSocket (`lib/websocket.ts`)

```typescript
import { wsManager } from '@/lib/websocket';

// Connect to training job
wsManager.connect(jobId, {
  onProgress: (progress) => {
    console.log('Progress:', progress);
  },
  onLog: (log) => {
    console.log('Log:', log);
  },
  onComplete: (status) => {
    console.log('Complete:', status);
  },
  onError: (error) => {
    console.error('Error:', error);
  }
});

// Disconnect
wsManager.disconnect();
```

### State Management (`store/pipelineStore.ts`)

```typescript
import { usePipelineStore } from '@/store/pipelineStore';

function MyComponent() {
  const { 
    modelInfo, 
    setModelInfo,
    currentStep,
    setCurrentStep 
  } = usePipelineStore();

  // Use state
  console.log(modelInfo);

  // Update state
  setModelInfo(newInfo);
}
```

## 🎨 Styling

### Tailwind Configuration

Custom colors defined in `tailwind.config.js`:

- **Primary**: Deep Purple (#7C3AED)
- **Secondary**: Electric Blue (#3B82F6)
- **Accent**: Cyan (#06B6D4)
- **Background**: Dark Slate (#0F172A)

### Custom Classes

```css
.text-gradient        /* Gradient text */
.glass                /* Glassmorphism effect */
.glass-card           /* Glass card with hover */
.animate-float        /* Floating animation */
.animate-gradient     /* Gradient shift animation */
```

## 📊 Data Flow

```
User Action
    ↓
Component Event
    ↓
Zustand Store Update
    ↓
API Client Call
    ↓
Backend API
    ↓
Response
    ↓
Store Update
    ↓
UI Re-render
```

For WebSocket:

```
Training Started
    ↓
WebSocket Connect
    ↓
Real-time Messages
    ↓
Store Updates
    ↓
UI Auto-refresh
```

## 🧪 Testing

### Manual Testing Checklist

- [ ] Landing page loads correctly
- [ ] Navigation works
- [ ] Model search and selection
- [ ] Dataset upload (drag & drop)
- [ ] Hyperparameter recommendations
- [ ] Training starts successfully
- [ ] Real-time updates display
- [ ] Charts render correctly
- [ ] Quantization options work
- [ ] Code generation works
- [ ] Downloads function properly

### Browser Support

- Chrome/Edge 90+
- Firefox 90+
- Safari 14+
- Mobile browsers

## 🐛 Troubleshooting

### Build Errors

```bash
# Clear cache and reinstall
rm -rf .next node_modules package-lock.json
npm install
```

### TypeScript Errors

```bash
# Run type check
npm run type-check

# Check tsconfig.json paths
```

### API Connection Issues

1. Verify backend is running on correct port
2. Check `.env.local` has correct API_URL
3. Check CORS settings in backend
4. Open browser console for network errors

### WebSocket Not Connecting

1. Verify WS_URL in `.env.local`
2. Check backend WebSocket endpoint
3. Look for connection errors in console
4. Ensure no proxy/firewall blocking WS

## 🚀 Deployment

### Vercel (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

Environment variables to set:
- `NEXT_PUBLIC_API_URL`: Your backend URL
- `NEXT_PUBLIC_WS_URL`: Your WebSocket URL

### Docker

```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
```

Build and run:

```bash
docker build -t llm-frontend .
docker run -p 3000:3000 llm-frontend
```

### Static Export

```bash
# Add to next.config.js
output: 'export'

# Build
npm run build

# Output in 'out/' directory
```

## 📝 Development Guidelines

### Code Style

- Use TypeScript for all files
- Follow ESLint configuration
- Use Tailwind for styling
- Component names in PascalCase
- Use 'use client' directive for client components

### Component Structure

```typescript
'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';

interface Props {
  title: string;
}

export function MyComponent({ title }: Props) {
  const [state, setState] = useState('');

  return (
    <div className="p-4">
      <h1>{title}</h1>
      {/* Component JSX */}
    </div>
  );
}
```

### State Management

- Use Zustand for global state
- Use React Query for server state
- Use useState for local component state
- Keep state close to where it's used

## 🤝 Contributing

1. Follow the existing code style
2. Add TypeScript types for all functions
3. Test on multiple browsers
4. Ensure responsive design
5. Add comments for complex logic

## 📞 Support

For issues and questions:
- Check backend logs
- Review browser console
- Verify API connectivity
- Check GitHub issues

---

**Built with ❤️ using Next.js 14, TypeScript, and Tailwind CSS**
