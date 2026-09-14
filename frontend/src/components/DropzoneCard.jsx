import { UploadCloud, X } from 'lucide-react';

/**
 * DropzoneCard component handling signature file uploads, previews, and removal.
 */
export function DropzoneCard({
  title,
  subtitle,
  file,
  preview,
  onFileSelect,
  onClear,
}) {
  return (
    <div className="glass-card" style={{ padding: '1.5rem' }}>
      <h3 className="card-label">{title}</h3>
      <div className={`dropzone ${file ? 'active' : ''}`}>
        <input
          type="file"
          accept="image/jpeg, image/png, image/jpg, image/webp"
          onChange={onFileSelect}
        />
        {preview ? (
          <>
            <img src={preview} alt={`${title} Preview`} className="image-preview" />
            <button
              className="btn-remove"
              onClick={(e) => {
                e.stopPropagation();
                onClear();
              }}
              title={`Remove ${title}`}
            >
              <X size={16} />
            </button>
          </>
        ) : (
          <>
            <UploadCloud className="dropzone-icon" />
            <div className="dropzone-title">Upload {title}</div>
            <div className="dropzone-desc">{subtitle}</div>
          </>
        )}
      </div>
    </div>
  );
}

export default DropzoneCard;
