export default function InputSection({ claim, evidence, onClaimChange, onEvidenceChange }) {
  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-semibold text-gray-700 mb-1.5">
          Luận điểm (Claim)
        </label>
        <textarea
          rows={3}
          value={claim}
          onChange={(e) => onClaimChange(e.target.value)}
          placeholder="Nhập luận điểm cần kiểm tra..."
          className="w-full px-3 py-2.5 text-sm border border-gray-200 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder-gray-400"
        />
      </div>
      <div>
        <label className="block text-sm font-semibold text-gray-700 mb-1.5">
          Bằng chứng (Evidence)
        </label>
        <textarea
          rows={5}
          value={evidence}
          onChange={(e) => onEvidenceChange(e.target.value)}
          placeholder="Nhập bằng chứng liên quan..."
          className="w-full px-3 py-2.5 text-sm border border-gray-200 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent placeholder-gray-400"
        />
      </div>
    </div>
  )
}
