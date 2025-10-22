"use client";

import { useEffect } from 'react';

/**
 * Component to clear all persisted storage on app mount
 * Ensures fresh state on every page load/reload
 */
export default function StorageCleaner() {
  useEffect(() => {
    // Clear any previous Zustand persisted state
    if (typeof window !== 'undefined') {
      // Remove old pipeline storage
      localStorage.removeItem('pipeline-storage');
      
      // Clear all items that might have been stored
      const keysToRemove: string[] = [];
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key?.includes('pipeline') || key?.includes('training') || key?.includes('model')) {
          keysToRemove.push(key);
        }
      }
      
      keysToRemove.forEach(key => localStorage.removeItem(key));
      
      console.log('✨ Session storage cleared - Fresh start!');
    }
  }, []);

  return null; // This component doesn't render anything
}
