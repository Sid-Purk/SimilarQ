import React, {useEffect, useState} from "react";

export default function AlertDialog({ open, message, onClose }){
    const [visible, setVisible] = useState(open);

  useEffect(() => {
    setVisible(open);
    if (open) {
      const timer = setTimeout(() => {
        setVisible(false);
        onClose();
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [open, onClose]);

  if (!visible) return null;
  return (
    <div className="fixed top-16 left-1/2 transform -translate-x-1/2 bg-lcred text-lcwhite px-8 py-4 rounded-2xl shadow-lg z-50 text-lg font-semibold border-4 border-lcgray">
      {message} <a target="_blank" className="text-blue-500" href="https://github.com/Sid-Purk/SimilarQ">here</a>
    </div>
  );
}