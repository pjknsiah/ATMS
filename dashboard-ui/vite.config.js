import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  root: '.',
  resolve: {
    alias: {
      '/web': '/web',
    },
  },
  server: {
    proxy: {
      // REST API → FastAPI on :8000
      '/api': 'http://localhost:8000',
      // WebSocket → FastAPI on :8000
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
})
