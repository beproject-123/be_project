export const Button = ({ children, onClick, className }) => {
  return (
    <button
      className={`px-4 py-2 rounded bg-green-500 text-white hover:bg-green-600 ${className}`}
      onClick={onClick}
    >
      {children}
    </button>
  );
};
