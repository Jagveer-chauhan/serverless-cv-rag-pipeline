import { User, Mail, Phone, MapPin, Briefcase, GraduationCap, Wrench, Award, Globe, CheckCircle } from 'lucide-react'

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

  const candidate = parsedData.candidate_info || {}
  const summary = parsedData.summary
  const experiences = parsedData.work_experience || []
  const educations = parsedData.education || []
  const skills = parsedData.skills || []
  const projects = parsedData.projects || []
  const certifications = parsedData.certifications || []

  return (
    <div className="space-y-6 text-xs text-slate-300">
      {/* 1. Candidate Header Card */}
      <div className="p-5 rounded-2xl bg-gradient-to-br from-slate-900 to-slate-850 border border-slate-800 relative overflow-hidden">
        <div className="flex items-start justify-between">
          <div className="flex items-start space-x-3.5">
            <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 font-bold text-lg font-mono">
              {candidate.name ? candidate.name.charAt(0) : <User className="w-6 h-6" />}
            </div>
            <div>
              <h2 className="text-base font-bold text-white tracking-tight">
                {candidate.name || 'Candidate Name'}
              </h2>
              {candidate.title && (
                <p className="text-xs text-emerald-400 font-medium mt-0.5">{candidate.title}</p>
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
                href={link}
                target="_blank"
                rel="noreferrer"
                className="px-2.5 py-1 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 hover:text-white flex items-center gap-1.5 transition-colors"
              >
                <Globe className="w-3 h-3 text-emerald-400" />
                <span className="truncate max-w-[200px]">{link.replace(/^https?:\/\//, '')}</span>
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
            {experiences.map((exp: any, idx: number) => (
              <div
                key={idx}
                className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2.5 relative"
              >
                <div className="flex items-start justify-between">
                  <div>
                    <h4 className="font-bold text-white text-xs">{exp.position}</h4>
                    <p className="text-emerald-400 font-semibold text-[11px]">{exp.company}</p>
                  </div>
                  <span className="px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 text-[10px] font-mono">
                    {exp.start_date || 'Past'} - {exp.is_current ? 'Present' : exp.end_date || 'Past'}
                  </span>
                </div>

                {exp.description && <p className="text-xs text-slate-300">{exp.description}</p>}

                {/* Key Achievements Bullets */}
                {exp.key_achievements && exp.key_achievements.length > 0 && (
                  <ul className="space-y-1.5 pl-1">
                    {exp.key_achievements.map((ach: string, aIdx: number) => (
                      <li key={aIdx} className="flex items-start space-x-2 text-slate-300">
                        <CheckCircle className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0 mt-0.5" />
                        <span>{ach}</span>
                      </li>
                    ))}
                  </ul>
                )}

                {/* Technologies tags */}
                {exp.technologies_used && exp.technologies_used.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 pt-2 border-t border-slate-800 font-mono text-[10px]">
                    {exp.technologies_used.map((tech: string, tIdx: number) => (
                      <span key={tIdx} className="px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                        {tech}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 4. Skills Grid */}
      {skills.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <Wrench className="w-4 h-4 text-cyan-400" />
            Skills &amp; Competencies
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {skills.map((cat: any, idx: number) => (
              <div key={idx} className="p-3.5 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2">
                <span className="text-[11px] font-bold text-slate-200 font-mono uppercase block">
                  {cat.category_name}
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {(cat.skills || []).map((s: string, sIdx: number) => (
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

      {/* 5. Projects, Education & Certifications */}
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

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {educations.length > 0 && (
          <div className="space-y-3">
            <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <GraduationCap className="w-4 h-4 text-purple-400" />
              Education
            </h3>
            <div className="space-y-2">
              {educations.map((edu: any, idx: number) => (
                <div key={idx} className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
                  <h4 className="font-semibold text-white text-xs">{edu.degree || 'Degree'}</h4>
                  <p className="text-purple-300 text-[11px]">{edu.institution}</p>
                  {edu.end_year && <p className="text-[10px] text-slate-500 font-mono">Class of {edu.end_year}</p>}
                </div>
              ))}
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
                  {cert.issuer && <p className="text-amber-300 text-[11px]">{cert.issuer}</p>}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
