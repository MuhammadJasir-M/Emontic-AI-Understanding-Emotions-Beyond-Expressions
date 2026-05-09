/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        base: '#0a0a0f',
        surface: '#11111b',
        card: '#16161f',
        'card-hover': '#1c1c28',
        violet: {
          DEFAULT: '#7c3aed',
          light: '#a855f7',
        },
        cyan: {
          DEFAULT: '#06b6d4',
          light: '#22d3ee',
        },
      },
      fontFamily: {
        sans: ['Rajdhani', 'Segoe UI', 'system-ui', 'sans-serif'],
      },
      borderRadius: {
        'xl': '20px',
        '2xl': '28px',
      },
    },
  },
  plugins: [],
}
