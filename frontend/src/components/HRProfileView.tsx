import { User, Mail, Phone, MapPin, Briefcase, GraduationCap, Wrench, Award, Globe, CheckCircle, FileText } from 'lucide-react'

interface HRProfileViewProps {
  parsedData: any
}

export function HRProfileView({ parsedData }: HRProfileViewProps) {
  if (!parsedData) {
    return (
      <div className="p-8 text-center text-xs text-slate-500 font-mono">
        No structured JSON profile data available.
      </div>
    )
  }

  // 1. Candidate Info
  const candidate = parsedData.candidate || parsedData.candidate_info || {}
  const summary = parsedData.summary || candidate.summary
  
  // 2. Experience normalization
  const rawExperiences = parsedData.experience || parsedData.work_experience || []
  const experiences = Array.isArray(rawExperiences) ? rawExperiences : []

  // 3. Education normalization
  const rawEducations = parsedData.education || []
  const educations = Array.isArray(rawEducations) ? rawEducations : []

  // 4. Skills normalization (handles dict, array of categories, or flat array)
  const rawSkills = parsedData.skills || {}
  const normalizedSkillCategories: { category_name: string; skills: string[] }[] = []

  if (Array.isArray(rawSkills)) {
    if (rawSkills.length > 0 && typeof rawSkills[0] === 'string') {
      normalizedSkillCategories.push({ category_name: 'Technical Skills', skills: rawSkills })
    } else {
      rawSkills.forEach((item: any) => {
        if (typeof item === 'string') {
          normalizedSkillCategories.push({ category_name: 'Skills', skills: [item] })
        } else if (item && item.category_name && Array.isArray(item.skills)) {
          normalizedSkillCategories.push({ category_name: item.category_name, skills: item.skills })
        }
      })
    }
  } else if (typeof rawSkills === 'object' && rawSkills !== null) {
    Object.entries(rawSkills).forEach(([catKey, val]) => {
      if (Array.isArray(val) && val.length > 0) {
        const formattedTitle = catKey
          .replace(/_/g, ' ')
          .replace(/\b\w/g, (c) => c.toUpperCase())
        normalizedSkillCategories.push({ category_name: formattedTitle, skills: val })
      }
    })
  }

  // 5. Projects and Custom Sections
  const projects = Array.isArray(parsedData.projects) ? parsedData.projects : []
  const sections = Array.isArray(parsedData.sections) ? parsedData.sections : []
  const certifications = Array.isArray(parsedData.certifications) ? parsedData.certifications : []

  return (
    <div className="space-y-6 text-xs text-slate-300">
      {/* 1. Candidate Header Card */}
      <div className="p-5 rounded-2xl bg-gradient-to-br from-slate-900 to-slate-850 border border-slate-800 relative overflow-hidden shadow-lg">
        <div className="flex items-start justify-between">
          <div className="flex items-start space-x-3.5">
            <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 font-bold text-lg font-mono flex-shrink-0">
              {candidate.name && candidate.name !== 'Candidate Name' ? (
                candidate.name.charAt(0).toUpperCase()
              ) : (
                <User className="w-6 h-6" />
              )}
            </div>
            <div>
              <h2 className="text-base font-bold text-white tracking-tight">
                {candidate.name || 'Candidate Name'}
              </h2>
              {(candidate.title || candidate.position) && (
                <p className="text-xs text-emerald-400 font-medium mt-0.5">
                  {candidate.title || candidate.position}
                </p>
              )}

              {/* Contact meta pills */}
              <div className="flex flex-wrap gap-3 mt-3 text-[11px] text-slate-400 font-mono">
                {candidate.email && (
                  <span className="flex items-center gap-1.5">
                    <Mail className="w-3.5 h-3.5 text-slate-500" />
                    {candidate.email}
                  </span>
                )}
                {candidate.phone && (
                  <span className="flex items-center gap-1.5">
                    <Phone className="w-3.5 h-3.5 text-slate-500" />
                    {candidate.phone}
                  </span>
                )}
                {candidate.location && (
                  <span className="flex items-center gap-1.5">
                    <MapPin className="w-3.5 h-3.5 text-slate-500" />
                    {candidate.location}
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Links */}
        {candidate.links && candidate.links.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-4 pt-3 border-t border-slate-800/80 font-mono text-[11px]">
            {candidate.links.map((link: string, idx: number) => (
              <a
                key={idx}
                href={link.startsWith('http') ? link : `https://${link}`}
                target="_blank"
                rel="noreferrer"
                className="px-2.5 py-1 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 hover:text-white flex items-center gap-1.5 transition-colors"
              >
                <Globe className="w-3 h-3 text-emerald-400" />
                <span className="truncate max-w-[220px]">{link.replace(/^https?:\/\//, '')}</span>
              </a>
            ))}
          </div>
        )}
      </div>

      {/* 2. Professional Summary */}
      {summary && (
        <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2">
          <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400">
            Professional Summary
          </h3>
          <p className="text-xs text-slate-200 leading-relaxed font-sans">{summary}</p>
        </div>
      )}

      {/* 3. Work Experience Timeline */}
      {experiences.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Briefcase className="w-4 h-4 text-emerald-400" />
            Work Experience ({experiences.length})
          </h3>

          <div className="space-y-3">
            {experiences.map((exp: any, idx: number) => {
              const roleTitle = exp.position || exp.title || 'Role'
              const companyName = exp.company || 'Company'
              const achievements = exp.key_achievements || exp.achievements || []
              const technologies = exp.technologies_used || exp.technologies || []
              const dateDisplay =
                exp.start_date || exp.end_date
                  ? `${exp.start_date || 'Past'} - ${exp.is_current ? 'Present' : exp.end_date || 'Past'}`
                  : null

              return (
                <div
                  key={idx}
                  className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2.5 relative"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <h4 className="font-bold text-white text-xs">{roleTitle}</h4>
                      <p className="text-emerald-400 font-semibold text-[11px]">{companyName}</p>
                    </div>
                    {dateDisplay && (
                      <span className="px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 text-[10px] font-mono">
                        {dateDisplay}
                      </span>
                    )}
                  </div>

                  {exp.description && (
                    <p className="text-xs text-slate-300 leading-relaxed">{exp.description}</p>
                  )}

                  {/* Achievements */}
                  {achievements.length > 0 && (
                    <ul className="space-y-1.5 pl-1">
                      {achievements.map((ach: string, aIdx: number) => (
                        <li key={aIdx} className="flex items-start space-x-2 text-slate-300">
                          <CheckCircle className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0 mt-0.5" />
                          <span>{ach}</span>
                        </li>
                      ))}
                    </ul>
                  )}

                  {/* Technologies */}
                  {technologies.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 pt-2 border-t border-slate-800 font-mono text-[10px]">
                      {technologies.map((tech: string, tIdx: number) => (
                        <span key={tIdx} className="px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                          {tech}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* 4. Skills Grid */}
      {normalizedSkillCategories.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Wrench className="w-4 h-4 text-cyan-400" />
            Skills &amp; Competencies
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {normalizedSkillCategories.map((cat, idx) => (
              <div key={idx} className="p-3.5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2">
                <span className="text-[11px] font-bold text-slate-200 font-mono uppercase block">
                  {cat.category_name}
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {cat.skills.map((s: string, sIdx: number) => (
                    <span
                      key={sIdx}
                      className="px-2 py-0.5 rounded-lg bg-emerald-500/10 text-emerald-300 border border-emerald-500/20 text-[11px]"
                    >
                      {s}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 5. Projects */}
      {projects.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Globe className="w-4 h-4 text-emerald-400" />
            Key Projects ({projects.length})
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {projects.map((proj: any, idx: number) => (
              <div key={idx} className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1.5">
                <h4 className="font-bold text-white text-xs">{proj.name}</h4>
                {proj.description && <p className="text-xs text-slate-300">{proj.description}</p>}
                {proj.technologies && proj.technologies.length > 0 && (
                  <div className="flex flex-wrap gap-1 pt-1 font-mono text-[10px]">
                    {proj.technologies.map((t: string, tIdx: number) => (
                      <span key={tIdx} className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
                        {t}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 6. Custom Sections (if present) */}
      {sections.length > 0 && (
        <div className="space-y-3">
          {sections.map((sec: any, idx: number) => (
            <div key={idx} className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2">
              <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <FileText className="w-4 h-4 text-cyan-400" />
                {sec.heading || `Section ${idx + 1}`}
              </h3>
              <p className="text-xs text-slate-300 whitespace-pre-line leading-relaxed">{sec.content}</p>
            </div>
          ))}
        </div>
      )}

      {/* 7. Education & Certifications */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {educations.length > 0 && (
          <div className="space-y-3">
            <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <GraduationCap className="w-4 h-4 text-purple-400" />
              Education
            </h3>
            <div className="space-y-2">
              {educations.map((edu: any, idx: number) => {
                const deg = edu.degree || 'Degree'
                const inst = edu.institution || 'Institution'
                const year = edu.end_year || edu.end_date || edu.start_year
                return (
                  <div key={idx} className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
                    <h4 className="font-semibold text-white text-xs">{deg}</h4>
                    <p className="text-purple-300 text-[11px]">{inst}</p>
                    {year && <p className="text-[10px] text-slate-500 font-mono">{year}</p>}
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {certifications.length > 0 && (
          <div className="space-y-3">
            <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <Award className="w-4 h-4 text-amber-400" />
              Certifications
            </h3>
            <div className="space-y-2">
              {certifications.map((cert: any, idx: number) => (
                <div key={idx} className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
                  <h4 className="font-semibold text-white text-xs">{cert.name}</h4>
                  {(cert.issuer || cert.date) && (
                    <p className="text-amber-300 text-[11px]">
                      {cert.issuer} {cert.date ? `(${cert.date})` : ''}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
