/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        heading: ['Rajdhani', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
        body: ['Inter', 'sans-serif'],
      },
      colors: {
        void: '#050505',
        panel: '#0A0A0A',
        surface: '#0F0F0F',
        neon: {
          cyan: '#00F0FF',
          red: '#FF2A6D',
          green: '#39FF14',
          amber: '#FFB800',
        }
      }
    }
  },
  plugins: [],
}
