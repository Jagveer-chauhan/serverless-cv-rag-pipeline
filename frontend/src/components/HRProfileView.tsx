import {
  User,
  Mail,
  Phone,
  MapPin,
  Briefcase,
  GraduationCap,
  Wrench,
  Award,
  Globe,
  CheckCircle,
  FileText,
  Calendar,
  Layers,
  Sparkles,
  Trophy
} from 'lucide-react'

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

  // 2. Experience normalization & bundling
  const rawExperiences = parsedData.experience || parsedData.work_experience || []
  const bundledExperiences: any[] = []

  if (Array.isArray(rawExperiences)) {
    rawExperiences.forEach((item: any) => {
      const title = (item.title || item.position || '').trim()
      const company = (item.company || '').trim()
      const isDummyCompany = company.toLowerCase() === 'company' || company === ''
      const isDummyTitle =
        !title ||
        title.toLowerCase() === 'role' ||
        title.toLowerCase() === 'professional' ||
        /^\(part\s+\d+\/\d+\)$/i.test(title)

      if ((isDummyCompany || isDummyTitle) && bundledExperiences.length > 0) {
        // Merge into previous experience bundle
        const prev = bundledExperiences[bundledExperiences.length - 1]
        const extraAch = item.achievements || item.key_achievements || []
        const extraTech = item.technologies || item.technologies_used || []
        if (item.description && !prev.description?.includes(item.description)) {
          prev.description = prev.description
            ? `${prev.description}\n${item.description}`
            : item.description
        }
        if (title && !isDummyTitle && !extraAch.includes(title)) {
          extraAch.unshift(title)
        }
        prev.achievements = Array.from(new Set([...(prev.achievements || []), ...extraAch]))
        prev.technologies = Array.from(new Set([...(prev.technologies || []), ...extraTech]))
        if (!prev.start_date && item.start_date) prev.start_date = item.start_date
        if (!prev.end_date && item.end_date) prev.end_date = item.end_date
        if (!prev.location && item.location) prev.location = item.location
      } else {
        // Parse location if contained in company string (e.g. "Forviz Mazars, Gurugram")
        let cleanCompany = company
        let loc = item.location || null
        if (!loc && company.includes(',')) {
          const parts = company.split(',').map((p: string) => p.trim())
          if (parts.length >= 2 && parts[parts.length - 1].length < 30) {
            loc = parts[parts.length - 1]
            cleanCompany = parts.slice(0, -1).join(', ')
          }
        }

        bundledExperiences.push({
          ...item,
          company: cleanCompany || company || 'Company',
          title: title || 'Role',
          location: loc,
          achievements: item.achievements || item.key_achievements || [],
          technologies: item.technologies || item.technologies_used || [],
        })
      }
    })
  }

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
    if (Array.isArray(rawSkills.explicit) && rawSkills.explicit.length > 0) {
      normalizedSkillCategories.push({ category_name: 'Core Skills', skills: rawSkills.explicit })
    }
    if (Array.isArray(rawSkills.inferred) && rawSkills.inferred.length > 0) {
      normalizedSkillCategories.push({ category_name: 'Inferred Skills', skills: rawSkills.inferred })
    }
    if (Array.isArray(rawSkills.soft_skills) && rawSkills.soft_skills.length > 0) {
      normalizedSkillCategories.push({ category_name: 'Soft Skills', skills: rawSkills.soft_skills })
    }

    Object.entries(rawSkills).forEach(([catKey, val]) => {
      if (['explicit', 'inferred', 'soft_skills'].includes(catKey)) return
      if (Array.isArray(val) && val.length > 0) {
        const formattedTitle = catKey
          .replace(/_/g, ' ')
          .replace(/\b\w/g, (c) => c.toUpperCase())
        normalizedSkillCategories.push({ category_name: formattedTitle, skills: val })
      }
    })
  }

  // 5. Projects and Custom Sections
  const rawProjects = parsedData.projects || []
  const rawSections = parsedData.sections || []

  // If projects are inside sections, separate them
  const projectList: any[] = Array.isArray(rawProjects) ? [...rawProjects] : []
  const customSectionList: any[] = []

  if (Array.isArray(rawSections)) {
    rawSections.forEach((sec: any) => {
      const heading = (sec.heading || '').toLowerCase()
      if (heading.includes('project') || heading.includes('system') || heading.includes('portal') || heading.includes('app')) {
        // If not already in project list
        if (!projectList.some((p: any) => (p.name || p.heading) === sec.heading)) {
          projectList.push({
            name: sec.heading,
            role: sec.role,
            description: sec.content,
            technologies: sec.technologies || []
          })
        }
      } else {
        customSectionList.push(sec)
      }
    })
  }

  // 6. Certifications & Training
  const certifications = Array.isArray(parsedData.certifications) ? parsedData.certifications : []

  // 7. Awards / Leadership Signals
  const rawAwards = parsedData.awards || []
  const awards = Array.isArray(rawAwards) ? rawAwards : []
  const leadershipSignals = parsedData.inferred?.leadership_signals || []

  // Helper to format structured project description lines
  const renderProjectContent = (desc: string) => {
    if (!desc) return null
    const lines = desc.split('\n').map((l) => l.trim()).filter(Boolean)
    const hasKeyValHeaders = lines.some((l) => /^[A-Za-z0-9\s&/-]{3,35}:\s+/.test(l))

    if (hasKeyValHeaders) {
      return (
        <div className="space-y-2 mt-2.5">
          {lines.map((l, i) => {
            const match = l.match(/^([A-Za-z0-9\s&/-]{3,35}):\s+(.*)/)
            if (match) {
              return (
                <div key={i} className="text-xs text-slate-300 pl-3 border-l-2 border-emerald-500/40 bg-slate-950/30 p-1.5 rounded-r-lg">
                  <span className="font-semibold text-emerald-300 font-mono text-[11px] block sm:inline">
                    {match[1]}:{' '}
                  </span>
                  <span className="text-slate-300 text-xs">{match[2]}</span>
                </div>
              )
            }
            return (
              <p key={i} className="text-xs text-slate-300 leading-relaxed">
                {l}
              </p>
            )
          })}
        </div>
      )
    }

    return <p className="text-xs text-slate-300 leading-relaxed mt-2 whitespace-pre-line">{desc}</p>
  }

  return (
    <div className="w-full space-y-6 text-xs text-slate-300">
      {/* 1. Candidate Header Card */}
      <div className="w-full p-5 rounded-2xl bg-gradient-to-br from-slate-900 via-slate-850 to-slate-900 border border-slate-800 relative overflow-hidden shadow-xl">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-start space-x-4">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-emerald-500/20 to-teal-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 font-bold text-xl font-mono flex-shrink-0 shadow-inner">
              {candidate.name && candidate.name !== 'Candidate Name' ? (
                candidate.name.charAt(0).toUpperCase()
              ) : (
                <User className="w-7 h-7" />
              )}
            </div>
            <div>
              <h2 className="text-lg font-bold text-white tracking-tight">
                {candidate.name || 'Candidate Name'}
              </h2>
              {(candidate.title || candidate.position) && (
                <p className="text-xs text-emerald-400 font-semibold mt-0.5 flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                  <span>{candidate.title || candidate.position}</span>
                </p>
              )}

              {/* Contact meta pills */}
              <div className="flex flex-wrap gap-2.5 mt-3 text-[11px] text-slate-400 font-mono">
                {candidate.email && (
                  <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-950/60 border border-slate-800">
                    <Mail className="w-3.5 h-3.5 text-emerald-400" />
                    {candidate.email}
                  </span>
                )}
                {candidate.phone && (
                  <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-950/60 border border-slate-800">
                    <Phone className="w-3.5 h-3.5 text-emerald-400" />
                    {candidate.phone}
                  </span>
                )}
                {candidate.location && (
                  <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-950/60 border border-slate-800">
                    <MapPin className="w-3.5 h-3.5 text-cyan-400" />
                    {candidate.location}
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Links */}
        {candidate.links && (Array.isArray(candidate.links) ? candidate.links.length > 0 : Object.keys(candidate.links).length > 0) && (
          <div className="flex flex-wrap gap-2 mt-4 pt-3 border-t border-slate-800 font-mono text-[11px]">
            {(Array.isArray(candidate.links) ? candidate.links : Object.values(candidate.links)).map((link: any, idx: number) => {
              if (!link || typeof link !== 'string') return null
              const url = link.startsWith('http') ? link : `https://${link}`
              return (
                <a
                  key={idx}
                  href={url}
                  target="_blank"
                  rel="noreferrer"
                  className="px-3 py-1 rounded-lg bg-slate-950/80 hover:bg-slate-800 border border-slate-800 text-slate-300 hover:text-white flex items-center gap-1.5 transition-colors"
                >
                  <Globe className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="truncate max-w-[260px]">{link.replace(/^https?:\/\//, '')}</span>
                </a>
              )
            })}
          </div>
        )}
      </div>

      {/* 2. Professional Summary */}
      {summary && (
        <div className="w-full p-5 rounded-2xl bg-slate-900/70 border border-slate-800 space-y-2.5 shadow-md">
          <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
            <FileText className="w-4 h-4 text-emerald-400" />
            Professional Summary
          </h3>
          <p className="text-xs text-slate-200 leading-relaxed font-sans">{summary}</p>
        </div>
      )}

      {/* 3. Work Experience (Complete Bundled Items) */}
      {bundledExperiences.length > 0 && (
        <div className="w-full space-y-3.5">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
              <Briefcase className="w-4 h-4 text-emerald-400" />
              Work Experience ({bundledExperiences.length})
            </h3>
          </div>

          <div className="space-y-4">
            {bundledExperiences.map((exp: any, idx: number) => {
              const roleTitle = exp.position || exp.title || 'Role'
              const companyName = exp.company || 'Company'
              const achievements = exp.key_achievements || exp.achievements || []
              const technologies = exp.technologies_used || exp.technologies || []
              const dateDisplay =
                exp.start_date || exp.end_date
                  ? `${exp.start_date || 'Past'} — ${exp.is_current ? 'Present' : exp.end_date || 'Present'}`
                  : null

              return (
                <div
                  key={idx}
                  className="w-full p-5 rounded-2xl bg-slate-900/80 border border-slate-800/90 hover:border-slate-700 space-y-3.5 relative transition-all shadow-md"
                >
                  <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-2 pb-3 border-b border-slate-800/80">
                    <div>
                      <h4 className="font-bold text-white text-sm tracking-tight">{roleTitle}</h4>
                      <div className="flex flex-wrap items-center gap-2 mt-1">
                        <span className="text-emerald-400 font-semibold text-xs">{companyName}</span>
                        {exp.location && (
                          <span className="flex items-center gap-1 text-[11px] text-slate-400 font-mono">
                            <MapPin className="w-3 h-3 text-slate-500" />
                            {exp.location}
                          </span>
                        )}
                      </div>
                    </div>
                    {dateDisplay && (
                      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-[11px] font-mono font-medium flex-shrink-0 self-start">
                        <Calendar className="w-3.5 h-3.5 text-emerald-400" />
                        {dateDisplay}
                      </span>
                    )}
                  </div>

                  {exp.description && (
                    <p className="text-xs text-slate-300 leading-relaxed">{exp.description}</p>
                  )}

                  {/* Achievements */}
                  {achievements.length > 0 && (
                    <div className="space-y-2">
                      <span className="text-[11px] font-mono uppercase text-slate-400 font-semibold block">
                        Key Responsibilities &amp; Achievements
                      </span>
                      <ul className="space-y-2 pl-0.5">
                        {achievements.map((ach: string, aIdx: number) => (
                          <li key={aIdx} className="flex items-start space-x-2.5 text-slate-300">
                            <CheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                            <span className="text-xs leading-relaxed">{ach}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Technologies */}
                  {technologies.length > 0 && (
                    <div className="flex flex-wrap items-center gap-1.5 pt-3 border-t border-slate-800/80 font-mono text-[10px]">
                      <span className="text-slate-400 mr-1 text-[11px]">Stack:</span>
                      {technologies.map((tech: string, tIdx: number) => (
                        <span
                          key={tIdx}
                          className="px-2 py-0.5 rounded-md bg-slate-950 border border-slate-800 text-slate-300 font-mono"
                        >
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

      {/* 4. Skills Section (100% Full Width) */}
      {normalizedSkillCategories.length > 0 && (
        <div className="w-full space-y-3.5">
          <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
            <Wrench className="w-4 h-4 text-cyan-400" />
            Skills &amp; Competencies
          </h3>

          <div className="w-full space-y-3">
            {normalizedSkillCategories.map((cat, idx) => (
              <div
                key={idx}
                className="w-full p-4 rounded-2xl bg-slate-900/70 border border-slate-800 space-y-2.5 shadow-sm"
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-100 font-mono uppercase tracking-wide flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-cyan-400 inline-block" />
                    {cat.category_name}
                  </span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-slate-800 text-slate-400">
                    {cat.skills.length} skills
                  </span>
                </div>
                <div className="flex flex-wrap gap-2 pt-1">
                  {cat.skills.map((s: string, sIdx: number) => (
                    <span
                      key={sIdx}
                      className="px-3 py-1 rounded-xl bg-gradient-to-r from-slate-950 to-slate-900 text-slate-200 border border-slate-700/60 hover:border-cyan-500/50 hover:text-cyan-300 text-xs font-medium transition-all shadow-sm"
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

      {/* 5. Key Projects (100% Full Width Distinct Highlighted Cards) */}
      {projectList.length > 0 && (
        <div className="w-full space-y-3.5">
          <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
            <Layers className="w-4 h-4 text-emerald-400" />
            Key Projects ({projectList.length})
          </h3>

          <div className="w-full space-y-4">
            {projectList.map((proj: any, idx: number) => {
              const projName = proj.name || proj.heading || `Project ${idx + 1}`
              const projRole = proj.role || null
              const projDesc = proj.description || proj.content || ''
              const projTech = proj.technologies || []

              return (
                <div
                  key={idx}
                  className="w-full p-5 rounded-2xl bg-gradient-to-br from-slate-900/90 via-slate-850 to-slate-900 border border-slate-800 hover:border-emerald-500/40 space-y-3 shadow-md transition-all"
                >
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2.5 border-b border-slate-800">
                    <div className="flex items-center gap-2.5">
                      <div className="w-8 h-8 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 font-mono font-bold text-xs flex-shrink-0">
                        {idx + 1}
                      </div>
                      <div>
                        <h4 className="font-bold text-white text-sm tracking-tight">{projName}</h4>
                        {projRole && (
                          <span className="text-emerald-400 text-xs font-medium font-mono">
                            {projRole}
                          </span>
                        )}
                      </div>
                    </div>
                    {projRole && !projName.includes(projRole) && (
                      <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-[11px] font-mono self-start sm:self-auto">
                        {projRole}
                      </span>
                    )}
                  </div>

                  {/* Project description / sub-sections */}
                  {renderProjectContent(projDesc)}

                  {/* Project Technologies */}
                  {projTech.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 pt-2 font-mono text-[10px]">
                      {projTech.map((t: string, tIdx: number) => (
                        <span key={tIdx} className="px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-slate-300">
                          {t}
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

      {/* 6. Education Section (100% Full Width) */}
      {educations.length > 0 && (
        <div className="w-full space-y-3.5">
          <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
            <GraduationCap className="w-4 h-4 text-purple-400" />
            Education ({educations.length})
          </h3>

          <div className="w-full space-y-3">
            {educations.map((edu: any, idx: number) => {
              const deg = edu.degree || 'Degree / Study'
              const inst = edu.institution || 'University / Institution'
              const dateRange =
                edu.start_date || edu.end_date || edu.start_year || edu.end_year
                  ? `${edu.start_date || edu.start_year || ''}${edu.start_date && edu.end_date && edu.start_date !== edu.end_date ? ' — ' : ''}${edu.end_date || edu.end_year || ''}`
                  : null

              return (
                <div
                  key={idx}
                  className="w-full p-4 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-purple-500/40 space-y-2 transition-all shadow-sm"
                >
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    <div>
                      <h4 className="font-bold text-white text-xs sm:text-sm">{deg}</h4>
                      <p className="text-purple-300 text-xs font-medium mt-0.5">{inst}</p>
                    </div>
                    {dateRange && (
                      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-300 text-[11px] font-mono font-medium self-start sm:self-auto">
                        <Calendar className="w-3.5 h-3.5 text-purple-400" />
                        {dateRange}
                      </span>
                    )}
                  </div>
                  {edu.notes && <p className="text-xs text-slate-400 pt-1">{edu.notes}</p>}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* 7. Certifications & Training (100% Full Width Dedicated Section) */}
      {certifications.length > 0 && (
        <div className="w-full space-y-3.5">
          <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
            <Award className="w-4 h-4 text-amber-400" />
            Certifications &amp; Training ({certifications.length})
          </h3>

          <div className="w-full space-y-3">
            {certifications.map((cert: any, idx: number) => {
              const certName = cert.name || cert.title || 'Certification'
              const issuer = cert.issuer || null
              const dateVal = cert.date || cert.issue_date || null
              const url = cert.url || cert.link || null

              return (
                <div
                  key={idx}
                  className="w-full p-4 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-amber-500/40 space-y-1.5 transition-all shadow-sm"
                >
                  <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-2">
                    <div className="space-y-1">
                      <h4 className="font-bold text-white text-xs">{certName}</h4>
                      {issuer && (
                        <p className="text-amber-300 text-[11px] font-semibold flex items-center gap-1.5">
                          <Award className="w-3 h-3 text-amber-400" />
                          <span>{issuer}</span>
                        </p>
                      )}
                    </div>
                    {dateVal && (
                      <span className="px-2.5 py-0.5 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-300 text-[10px] font-mono self-start">
                        {dateVal}
                      </span>
                    )}
                  </div>
                  {url && (
                    <a
                      href={url}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1 text-[11px] text-amber-400/80 hover:text-amber-300 font-mono underline pt-1"
                    >
                      <Globe className="w-3 h-3" />
                      View Credential
                    </a>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* 8. Awards & Honors (if present) */}
      {awards.length > 0 && (
        <div className="w-full space-y-3.5">
          <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
            <Trophy className="w-4 h-4 text-amber-400" />
            Honors &amp; Awards ({awards.length})
          </h3>
          <div className="w-full space-y-2.5">
            {awards.map((award: any, idx: number) => {
              const aName = typeof award === 'string' ? award : award.name || award.title || 'Award'
              const aIssuer = typeof award === 'object' ? award.issuer || award.organization : null
              const aDate = typeof award === 'object' ? award.date || award.year : null
              return (
                <div
                  key={idx}
                  className="w-full p-4 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-1.5 shadow-sm"
                >
                  <div className="flex items-start justify-between gap-2">
                    <h4 className="font-bold text-white text-xs">{aName}</h4>
                    {aDate && (
                      <span className="px-2.5 py-0.5 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-300 text-[10px] font-mono">
                        {aDate}
                      </span>
                    )}
                  </div>
                  {aIssuer && <p className="text-amber-300 text-[11px]">{aIssuer}</p>}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* 9. Additional Custom Sections */}
      {customSectionList.length > 0 && (
        <div className="w-full space-y-3.5">
          {customSectionList.map((sec: any, idx: number) => (
            <div key={idx} className="w-full p-4 rounded-2xl bg-slate-900/70 border border-slate-800 space-y-2 shadow-sm">
              <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <FileText className="w-4 h-4 text-cyan-400" />
                {sec.heading || `Section ${idx + 1}`}
              </h3>
              <p className="text-xs text-slate-300 whitespace-pre-line leading-relaxed">{sec.content}</p>
            </div>
          ))}
        </div>
      )}

      {/* 10. Leadership Signals / Inferred Insights (if available) */}
      {leadershipSignals.length > 0 && (
        <div className="w-full p-4 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-2 shadow-sm">
          <h3 className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
            <Trophy className="w-4 h-4 text-amber-400" />
            Leadership &amp; Ownership Signals
          </h3>
          <ul className="space-y-1.5 pl-1">
            {leadershipSignals.map((sig: string, idx: number) => (
              <li key={idx} className="flex items-start space-x-2 text-slate-300 text-xs">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400 mt-1.5 flex-shrink-0" />
                <span>{sig}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
