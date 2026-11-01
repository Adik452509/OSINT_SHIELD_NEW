import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The API runs separately: python scripts/10_serve.py (http://127.0.0.1:8000).
// Its CORS settings (configs/serve.yaml) allow this dev server's port, 5173.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true },
})
