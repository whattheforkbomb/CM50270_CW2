%%
%% This is file `bathx.bst',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% bath-bst.dtx  (with options: `bst2')
%% ----------------------------------------------------------------
%% bath-bst --- Harvard referencing style as recommended by the University of Bath Library
%% Author:  Alex Ball
%% E-mail:  a.j.ball@bath.ac.uk
%% License: Released under the LaTeX Project Public License v1.3c or later
%% See:     http://www.latex-project.org/lppl.txt
%% ----------------------------------------------------------------
%% 
ENTRY
  { address
    archive
    author
    booktitle
    casenumber
    chapter
    doi
    edition
    editor
    eid
    entrysubtype
    eprint
    eventyear
    howpublished
    institution
    journal
    key
    keywords
    language
    library
    month
    note
    number
    options
    organization
    pages
    publisher
    pubstate
    school
    series
    sortyear
    title
    titleaddon
    translator
    type
    url
    urldate
    urlyear
    userb
    venue
    version
    volume
    year
  }
  {}
  { label extra.label sort.label short.list }
INTEGERS { output.state before.all mid.clause mid.sentence after.sentence after.block }
FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.clause :=
  #2 'mid.sentence :=
  #3 'after.sentence :=
  #4 'after.block :=
}
STRINGS { s t }
FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =
    { ", " * write$ }
    { output.state after.sentence =
        { add.period$ " " * write$ }
        { output.state before.all =
            'write$
            { output.state mid.clause =
                { " " * write$ }
                { add.period$ write$
                  newline$
                  "\newblock " write$
                }
              if$
            }
          if$
        }
      if$
      mid.sentence 'output.state :=
    }
  if$
  s
}
FUNCTION {end.clause}
{ output.state mid.clause =
    { mid.sentence 'output.state := }
    'skip$
  if$
}
FUNCTION {output}
{ duplicate$ empty$
    { pop$ end.clause }
    'output.nonnull
  if$
}
FUNCTION {output.check}
{ 't :=
  duplicate$ empty$
    { pop$ "empty " t * " in " * cite$ * warning$
      end.clause
    }
    'output.nonnull
  if$
}
FUNCTION {new.block}
{ output.state before.all =
    'skip$
    { after.block 'output.state := }
  if$
}
FUNCTION {new.sentence}
{ output.state after.block =
    'skip$
    { output.state before.all =
        'skip$
        { after.sentence 'output.state := }
      if$
    }
  if$
}
FUNCTION {continue.clause}
{ output.state after.sentence =
    'skip$
    { output.state after.block =
        'skip$
        { output.state before.all =
            'skip$
            { mid.clause 'output.state := }
          if$
        }
      if$
    }
  if$
}
FUNCTION {date.block}
{ new.block
}
FUNCTION {bibinfo.check}
{ swap$
  duplicate$ missing$
    { pop$ pop$
      ""
    }
    { duplicate$ empty$
        { swap$ pop$
        }
        { swap$
          "\bibinfo{" swap$ * "}{" * swap$ * "}" *
        }
      if$
    }
  if$
}
FUNCTION {bibinfo.warn}
{ swap$
  duplicate$ missing$
    { swap$ "missing " swap$ * " in " * cite$ * warning$ pop$
      ""
    }
    { duplicate$ empty$
        { swap$ "empty " swap$ * " in " * cite$ * warning$
        }
        { swap$
          "\bibinfo{" swap$ * "}{" * swap$ * "}" *
        }
      if$
    }
  if$
}
FUNCTION {fin.entry}
{ add.period$
  write$
  newline$
}

FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}
FUNCTION {and}
{   'skip$
    { pop$ #0 }
  if$
}
FUNCTION {or}
{   { pop$ #1 }
    'skip$
  if$
}
FUNCTION {new.block.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.block
  if$
}
FUNCTION {field.or.null}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}
FUNCTION {emphasize}
{ duplicate$ empty$
    { pop$ "" }
    { "\emph{" swap$ * "}" * }
  if$
}
FUNCTION {tie.or.space.prefix}
{ duplicate$ text.length$ #3 <
    { "~" }
    { " " }
  if$
  swap$
}
FUNCTION {capitalize}
{ "u" change.case$ "t" change.case$ }
FUNCTION {space.word}
{ " " swap$ * " " * }
FUNCTION {either.or.check}
{ empty$
    'pop$
    { "can't use both " swap$ * " fields in " * cite$ * warning$ }
  if$
}
FUNCTION {is.online}
{ url empty$ doi empty$ and
    { type$ "online" =
        { #1 }
        { #0 }
      if$
    }
    { #1 }
  if$
}
FUNCTION {bbl.and}
{ "and"}

FUNCTION {bbl.byeditor}
{ "Ed. by" }

FUNCTION {bbl.etal}
{ "et~al." }

FUNCTION {bbl.editors}
{ "eds" }

FUNCTION {bbl.editor}
{ "ed." }

FUNCTION {bbl.edby}
{ "edited by" }

FUNCTION {bbl.translator}
{ "Trans." }

FUNCTION {bbl.edition}
{ "ed." }

FUNCTION {bbl.volume}
{ "vol." }

FUNCTION {bbl.of}
{ "of" }

FUNCTION {bbl.number}
{ "no." }

FUNCTION {bbl.nr}
{ "no." }

FUNCTION {bbl.in}
{ "in" }

FUNCTION {bbl.pages}
{ "pp." }

FUNCTION {bbl.page}
{ "p." }

FUNCTION {bbl.chapter}
{ "chap." }

FUNCTION {bbl.techrep}
{ "Tech. Rep." }

FUNCTION {bbl.mthesis}
{ "Master's thesis" }

FUNCTION {bbl.phdthesis}
{ "Ph.D. thesis" }

FUNCTION {bbl.first}
{ "1st" }

FUNCTION {bbl.second}
{ "2nd" }

FUNCTION {bbl.third}
{ "3rd" }

FUNCTION {bbl.fourth}
{ "4th" }

FUNCTION {bbl.fifth}
{ "5th" }

FUNCTION {bbl.st}
{ "st" }

FUNCTION {bbl.nd}
{ "nd" }

FUNCTION {bbl.rd}
{ "rd" }

FUNCTION {bbl.th}
{ "th" }

FUNCTION {bbl.commission}
{ "Commission Decision" }

FUNCTION {bbl.online}
{ "[Online]" }

FUNCTION {bbl.nodate}
{ "n.d." }

FUNCTION {bbl.inpreparation}
{ "preprint" }

FUNCTION {bbl.submitted}
{ "preprint" }

FUNCTION {bbl.inpress}
{ "in press" }

FUNCTION {bbl.unpublished}
{ "Unpublished" }

MACRO {jan} {"January"}

MACRO {feb} {"February"}

MACRO {mar} {"March"}

MACRO {apr} {"April"}

MACRO {may} {"May"}

MACRO {jun} {"June"}

MACRO {jul} {"July"}

MACRO {aug} {"August"}

MACRO {sep} {"September"}

MACRO {oct} {"October"}

MACRO {nov} {"November"}

MACRO {dec} {"December"}

FUNCTION {eng.ord}
{ duplicate$ "1" swap$ *
  #-2 #1 substring$ "1" =
     { bbl.th * }
     { duplicate$ #-1 #1 substring$
       duplicate$ "1" =
         { pop$ bbl.st * }
         { duplicate$ "2" =
             { pop$ bbl.nd * }
             { "3" =
                 { bbl.rd * }
                 { bbl.th * }
               if$
             }
           if$
          }
       if$
     }
   if$
}

INTEGERS { nameptr namesleft numnames }
STRINGS  { bibinfo }
FUNCTION {format.names}
{ 'bibinfo :=
  duplicate$ empty$
    'skip$
    { 's :=
      "" 't :=
      #1 'nameptr :=
      s num.names$ 'numnames :=
      numnames 'namesleft :=
        { namesleft #0 > }
        { s nameptr
          "{ll}{, jj}{, f{.}.}{~vv}"
          format.name$
          bibinfo bibinfo.check
          't :=
          nameptr #1 >
            { namesleft #1 >
                { ", " * t * }
                { s nameptr "{ll}" format.name$ duplicate$ "others" =
                    { 't := }
                    { pop$ }
                  if$
                  t "others" =
                    { " " * bbl.etal *
                    }
                    { bbl.and
                      space.word * t *
                    }
                  if$
                }
              if$
            }
            't
          if$
          nameptr #1 + 'nameptr :=
          namesleft #1 - 'namesleft :=
        }
      while$
    }
  if$
}
FUNCTION {format.names.ed}
{ 'bibinfo :=
  duplicate$ empty$
    'skip$
    { 's :=
      "" 't :=
      #1 'nameptr :=
      s num.names$ 'numnames :=
      numnames 'namesleft :=
        { namesleft #0 > }
        { s nameptr
          "{f{.}.~}{vv~}{ll}{ jj}"
          format.name$
          bibinfo bibinfo.check
          't :=
          nameptr #1 >
            { namesleft #1 >
                { ", " * t * }
                { s nameptr "{ll}" format.name$ duplicate$ "others" =
                    { 't := }
                    { pop$ }
                  if$
                  t "others" =
                    { " " * bbl.etal *
                    }
                    { bbl.and
                      space.word * t *
                    }
                  if$
                }
              if$
            }
            't
          if$
          nameptr #1 + 'nameptr :=
          namesleft #1 - 'namesleft :=
        }
      while$
    }
  if$
}
FUNCTION {format.authors}
{ author "author" format.names
}
FUNCTION {format.byeditors}
{ editor "editor" format.names.ed duplicate$ empty$ 'skip$
    { bbl.byeditor
      " " *
      swap$ *
    }
  if$
}
FUNCTION {get.bbl.editor}
{ editor num.names$ #1 > 'bbl.editors 'bbl.editor if$ }
FUNCTION {format.editors}
{ editor "editor" format.names duplicate$ empty$ 'skip$
    { "," *
      " " *
      get.bbl.editor
      *
    }
  if$
}
FUNCTION {format.translators}
{ translator "translator" format.names.ed duplicate$ empty$ 'skip$
    { "," *
      " " *
      bbl.translator *
      "(" swap$ *
      ")" *
    }
  if$
}
FUNCTION {format.full.names}
{'s :=
 "" 't :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr
      "{vv~}{ll}" format.name$
      't :=
      nameptr #1 >
        { namesleft #1 >
            { ", " * t * }
            { s nameptr "{ll}" format.name$ duplicate$ "others" =
                { 't := }
                { pop$ }
              if$
              t "others" =
                { " " * bbl.etal *
                }
                { bbl.and
                  space.word * t *
                }
              if$
            }
          if$
        }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}
FUNCTION {author.editor.key.full}
{ author empty$
    { editor empty$
        { key empty$
            { cite$ #1 #3 substring$ }
            'key
          if$
        }
        { editor format.full.names }
      if$
    }
    { author format.full.names }
  if$
}
FUNCTION {editor.key.full}
{ editor empty$
    { key empty$
         { cite$ #1 #3 substring$ }
          'key
      if$
    }
    { editor format.full.names }
  if$
}
FUNCTION {author.key.full}
{ author empty$
    { key empty$
         { cite$ #1 #3 substring$ }
          'key
      if$
    }
    { author format.full.names }
  if$
}
FUNCTION {make.full.names}
{ type$ "book" =
  type$ "inbook" =
  or
    'author.editor.key.full
    { type$ "proceedings" =
        'editor.key.full
        'author.key.full
      if$
    }
  if$
}
FUNCTION {output.bibitem}
{ newline$
  "\bibitem[{" write$
  label write$
  ")" make.full.names duplicate$ short.list =
     { pop$ }
     { * }
   if$
  "}]{" * write$
  cite$ write$
  "}" write$
  newline$
  ""
  before.all 'output.state :=
}
FUNCTION {format.key}
{ empty$
    { key field.or.null }
    { "" }
  if$
}
FUNCTION {select.language}
{ duplicate$ empty$
    'pop$
    { language empty$
        'skip$
        { "{\selectlanguage{" language * "}" * swap$ * "}" * }
      if$
    }
  if$
}
FUNCTION {format.title}
{ title
  duplicate$ empty$ 'skip$
    { "t" change.case$ }
  if$
  "title" bibinfo.check
  duplicate$ empty$
    'skip$
    { select.language }
  if$
}
FUNCTION {format.btitle}
{ title
  duplicate$ empty$ 'skip$
    { "t" change.case$ }
  if$
  "title" bibinfo.check
  duplicate$ empty$ 'skip$
    { emphasize
      select.language
    }
  if$
}
FUNCTION {format.titleaddon}
{ titleaddon
  duplicate$ empty$ 'skip$
    { "[" swap$ "titleaddon" bibinfo.check * "]" * }
  if$
}
FUNCTION {format.version}
{ version
  duplicate$ empty$ 'skip$
    { "(v." swap$ *
      "version" bibinfo.check
      ")" *
    }
  if$
}
FUNCTION {get.title.addenda}
{ continue.clause
  format.version output
  continue.clause
  format.titleaddon output
  continue.clause
  is.online
  journal empty$
  and
    booktitle empty$
      type$ "inreference" =
      author empty$
      and
    or
  and
    library empty$
    type$ "image" =
    or
  and
  type$ "software" = not
  and
    type empty$
      type$ "booklet" = not
      type$ "audio" = not
      and
      type$ "video" = not
      and
      type$ "music" = not
      and
      type$ "movie" = not
      and
    or
  and
    { bbl.online output }
    'skip$
  if$
  continue.clause
  type$ "reference" =
    { format.byeditors output }
    'skip$
  if$
  format.translators output
}
FUNCTION {word.in}
{ bbl.in capitalize
  ":" *
  " " * }
FUNCTION {format.label.booktitle}
{ booktitle
  duplicate$ empty$ 'skip$
    { "t" change.case$ }
  if$
  "booktitle" bibinfo.check
}
FUNCTION {format.booktitle}
{ format.label.booktitle
  emphasize
}
FUNCTION {format.in.ed.booktitle}
{ format.booktitle
  duplicate$ empty$
    'skip$
    {
      is.online
        { " " *
          bbl.online *
        }
        'skip$
      if$
      editor "editor" format.names.ed duplicate$ empty$ 'pop$
        { "," *
          " " *
          get.bbl.editor add.period$
          " " *
          * swap$
          *
          word.in swap$ *
        }
      if$
    }
  if$
}
FUNCTION {legal.journal.shape}
{ duplicate$ "OJ" =
    'emphasize
    'skip$
  if$
}
FUNCTION {nodate.check}
{
duplicate$ empty$
  sortyear empty$
  and
    { options field.or.null "nonodate" =
        'skip$
        { pop$ bbl.nodate }
      if$
    }
    'skip$
  if$
}
FUNCTION {format.date}
{ year "year" bibinfo.check nodate.check
  extra.label *
}
FUNCTION {format.date.brackets}
{ format.date
  "[" swap$ * "]" *
}
FUNCTION {format.date.parens}
{ format.date
  "(" swap$ * ")" *
}
FUNCTION {format.jur.date.essential}
{ options field.or.null "scottish-style" =
  keywords field.or.null "sc" =
  or
    'format.date
    'format.date.brackets
  if$
}
FUNCTION {format.jur.date}
{ journal field.or.null 's :=
  s "OJ" =
  s "ECR" =
  or
    'format.date.brackets
    { options field.or.null "year-essential=false" =
        'format.date.parens
        { volume empty$
            'format.jur.date.essential
            { options field.or.null "year-essential=true" =
                'format.jur.date.essential
                'format.date.parens
              if$
            }
          if$
        }
      if$
    }
  if$
}
FUNCTION {format.eu.date}
{ journal field.or.null 's :=
  s "OJ" =
  s "ECR" =
  or
    'format.date.brackets
    'format.date
  if$
}
FUNCTION {format.bvolume}
{ volume empty$
    { "" }
    { bbl.volume volume tie.or.space.prefix
      "volume" bibinfo.check * *
      series "series" bibinfo.check
      duplicate$ empty$ 'pop$
        { emphasize ", " * swap$ * }
      if$
      "volume and number" number either.or.check
    }
  if$
}
FUNCTION {format.number.series}
{ volume empty$
    { number empty$
        { series field.or.null }
        { series empty$
            { number "number" bibinfo.check }
            { series "series" bibinfo.check
              number tie.or.space.prefix "number" bibinfo.check * *
            }
          if$
        }
      if$
    }
    { "" }
  if$
}
FUNCTION {is.num}
{ chr.to.int$
  duplicate$ "0" chr.to.int$ < not
  swap$ "9" chr.to.int$ > not and
}
FUNCTION {extract.num}
{ duplicate$ 't :=
  "" 's :=
  { t empty$ not }
  { t #1 #1 substring$
    t #2 global.max$ substring$ 't :=
    duplicate$ is.num
      { s swap$ * 's := }
      { pop$ "" 't := }
    if$
  }
  while$
  s empty$
    'skip$
    { pop$ s }
  if$
}
FUNCTION {convert.edition}
{ extract.num "l" change.case$ 's :=
  s "first" = s "1" = or
    { bbl.first 't := }
    { s "second" = s "2" = or
        { bbl.second 't := }
        { s "third" = s "3" = or
            { bbl.third 't := }
            { s "fourth" = s "4" = or
                { bbl.fourth 't := }
                { s "fifth" = s "5" = or
                    { bbl.fifth 't := }
                    { s #1 #1 substring$ is.num
                        { s eng.ord 't := }
                        { edition 't := }
                      if$
                    }
                  if$
                }
              if$
            }
          if$
        }
      if$
    }
  if$
  t
}
FUNCTION {format.edition}
{ edition duplicate$ empty$ 'skip$
    { convert.edition
      output.state mid.sentence =
        { "l" }
        { "t" }
      if$ change.case$
      "edition" bibinfo.check
      " " * bbl.edition *
    }
  if$
}
FUNCTION {format.thesis.type}
{ type duplicate$ empty$
    'pop$
    { swap$ pop$
      "t" change.case$ "type" bibinfo.check
    }
  if$
}
FUNCTION {format.booklet.type}
{ type duplicate$ empty$
    { pop$ "" }
    { "type" bibinfo.check
      is.online author empty$ and editor empty$ and
        { " " * bbl.online * }
        'skip$
      if$
    }
  if$
}
FUNCTION {add.cmd.number}
{ ". " swap$ "number" bibinfo.check *
}
FUNCTION {format.legr.number}
{ series field.or.null 's :=
  s "C" =
    'add.cmd.number
    { s "Cd" =
        'add.cmd.number
        { s "Cmd" =
            'add.cmd.number
            { s "Cmnd" =
                'add.cmd.number
                { s "Cm" =
                    'add.cmd.number
                    { s #1 #3 substring$ 't :=
                      t "HL " =
                        { ", (" swap$
                          "number" bibinfo.check *
                          ")" *
                        }
                        { ", " swap$
                          "number" bibinfo.check *
                        }
                      if$
                    }
                  if$
                }
              if$
            }
          if$
        }
      if$
    }
  if$
}
FUNCTION {format.tr.number}
{ series duplicate$ empty$
    { pop$
      type duplicate$ empty$
        { pop$
          number duplicate$ empty$
            { pop$ ""
            }
            { "(" swap$
              "number" bibinfo.check
              * ")" *
            }
          if$
        }
        { "(" swap$
          "t" change.case$ "type" bibinfo.check
          *
          number duplicate$ empty$
            'pop$
            { tie.or.space.prefix
              "number" bibinfo.check
              * *
            }
          if$
          ")" *
        }
      if$
    }
    { "(" swap$ *
      type duplicate$ empty$
        { pop$
          number duplicate$ empty$
            'pop$
            { format.legr.number * }
          if$
        }
        { "t" change.case$ "type" bibinfo.check
          *
          number duplicate$ empty$
            'pop$
            { tie.or.space.prefix
              "number" bibinfo.check
              * *
            }
          if$
        }
      if$
      ")" *
      continue.clause
    }
  if$
}
FUNCTION {format.manual.number}
{ series duplicate$ empty$
    { pop$
      type duplicate$ empty$
        { pop$
          number duplicate$ empty$
            { pop$ ""
            }
            { "number" bibinfo.check
            }
          if$
        }
        { "t" change.case$ "type" bibinfo.check
          number duplicate$ empty$
            'pop$
            { tie.or.space.prefix
              "number" bibinfo.check
              * *
            }
          if$
        }
      if$
    }
    { number duplicate$ empty$
        'pop$
        { tie.or.space.prefix
          "number" bibinfo.check
          * *
        }
      if$
      type duplicate$ empty$
        'pop$
        { ", " swap$ *
          "t" change.case$ "type" bibinfo.check
          *
        }
      if$
    }
  if$
}
FUNCTION {format.chapter}
{ chapter duplicate$ empty$
    { pop$ "" }
    { "chapter" bibinfo.check
      "c." swap$ *
    }
  if$
}
FUNCTION {format.series.number.chapter}
{ series duplicate$ empty$
    { pop$ "(" }
    { "series" bibinfo.check
      "(" swap$ *
    }
  if$
  type duplicate$ empty$
    { pop$
      number duplicate$ empty$
        'pop$
        { "number" bibinfo.check
          swap$ duplicate$ "(" =
            { swap$ * }
            { ", " * swap$ * }
          if$
        }
      if$
      chapter duplicate$ empty$
        'pop$
        { "chapter" bibinfo.check
          swap$ duplicate$ "(" =
            { swap$ * }
            { ", c." * swap$ * }
          if$
        }
      if$
    }
    { "type" bibinfo.check
      swap$ duplicate$ "(" =
        { swap$ * }
        { ", " * swap$ * }
      if$
      number duplicate$ empty$
        { pop$
          chapter duplicate$ empty$
            'pop$
            { "chapter" bibinfo.check
              " c." swap$ * *
            }
          if$
        }
        { "number" bibinfo.check
          " " swap$ * *
          chapter duplicate$ empty$
            'pop$
            { "chapter" bibinfo.check
              ", c." swap$ * *
            }
          if$
        }
      if$
    }
  if$
  ")" *
  duplicate$ "()" =
    { pop$ "" }
    'skip$
  if$
}
FUNCTION {format.case.number}
{ casenumber duplicate$ empty$
    { pop$
      number duplicate$ empty$
        { pop$ "" }
        { institution field.or.null "Commission" =
          userb empty$ not
          and
            { userb "userb" bibinfo.check
              "(" swap$ * ") " * bbl.commission * " " *
              swap$ "number" bibinfo.check *
            }
            { "number" bibinfo.check
            "(" swap$ * ")" *
            }
          if$
        }
      if$
    }
    { "casenumber" bibinfo.check
      "(" swap$ * ")" *
      institution field.or.null "Commission" =
      number empty$ not
      and
        { " " * bbl.commission * " " *
          number "number" bibinfo.check *
        }
        'skip$
      if$
    }
  if$
}
FUNCTION {eu.case.check}
{ journal field.or.null "OJ" =
  number empty$ not
  or
  casenumber empty$ not
  or
    'continue.clause
    'new.block
  if$
}
FUNCTION {format.note}
{ note empty$
    { "" }
    { note #1 #1 substring$
      duplicate$ "{" =
        'skip$
        { output.state mid.sentence =
          output.state mid.clause =
          or
            { "l" }
            { "u" }
          if$
          change.case$
        }
      if$
      note #2 global.max$ substring$ * "note" bibinfo.check
    }
  if$
}
FUNCTION {format.org.or.pub}
{ 't :=
  ""
  address empty$ t empty$ and
    'skip$
    { address "address" bibinfo.check *
      t empty$
        'skip$
        { address empty$
            'skip$
            { ": " * }
          if$
          t *
        }
      if$
    }
  if$
}
FUNCTION {format.publisher.address}
{ publisher "publisher" bibinfo.warn format.org.or.pub
}
FUNCTION {format.organization.address}
{ publisher empty$
    { organization "organization" bibinfo.check format.org.or.pub }
    { publisher "publisher" bibinfo.warn format.org.or.pub
      organization duplicate$ empty$
        'pop$
        { "organization" bibinfo.check ". " * swap$ * }
      if$
    }
  if$
}
FUNCTION {format.institution.address}
{ publisher empty$
    { institution "institution" bibinfo.check format.org.or.pub }
    { publisher "publisher" bibinfo.warn format.org.or.pub
      institution duplicate$ empty$
        'pop$
        { "institution" bibinfo.check ". " * swap$ * }
      if$
    }
  if$
}
FUNCTION {format.library}
{ library "library" bibinfo.check
  emphasize
  duplicate$ empty$ not
  is.online
  and
  journal empty$
  and
    { " " * bbl.online * }
    'skip$
  if$
}
FUNCTION {format.img.library}
{ library duplicate$ empty$
    { pop$ "" }
    { "library" bibinfo.check
      "At: " swap$ *
    }
  if$
}
FUNCTION {format.img.library.address}
{ address empty$ library empty$ and institution empty$ and
    { "" }
    { "At: "
      address "address" bibinfo.check *
      library duplicate$ empty$
        { institution duplicate$ empty$
            { pop$ "" }
            { address empty$
              'skip$
              { ". " swap$ }
              if$
              "institution" bibinfo.check * *
            }
          if$
        }
        { address empty$
            'skip$
            { ". " swap$ }
          if$
          "library" bibinfo.check * *
        }
      if$
    }
  if$
}
FUNCTION {format.pub.org.lib.address}
{ publisher duplicate$ empty$
  { pop$
    organization duplicate$ empty$
      { pop$
        format.img.library.address }
      { "organization" bibinfo.check
        library empty$
          { format.org.or.pub }
          { ". " *
            format.img.library.address *
          }
        if$
      }
    if$
  }
  { "publisher" bibinfo.warn format.org.or.pub
    library empty$
      'skip$
      { ". " *
        format.img.library *
      }
    if$
  }
  if$
}

INTEGERS { multiresult }
FUNCTION {multi.page.check}
{ 't :=
  #0 'multiresult :=
    { multiresult not
      t empty$ not
      and
    }
    { t #1 #1 substring$
      duplicate$ "-" =
      swap$ duplicate$ "," =
      swap$ "+" =
      or or
        { #1 'multiresult := }
        { t #2 global.max$ substring$ 't := }
      if$
    }
  while$
  multiresult
}
FUNCTION {n.dashify}
{ 't :=
  ""
    { t empty$ not }
    { t #1 #1 substring$ "-" =
        { t #1 #2 substring$ "--" = not
            { "--" *
              t #2 global.max$ substring$ 't :=
            }
            {   { t #1 #1 substring$ "-" = }
                { "-" *
                  t #2 global.max$ substring$ 't :=
                }
              while$
            }
          if$
        }
        { t #1 #1 substring$ *
          t #2 global.max$ substring$ 't :=
        }
      if$
    }
  while$
}
FUNCTION {format.pages}
{ pages duplicate$ empty$ 'skip$
    { duplicate$ multi.page.check
        { bbl.pages swap$
          n.dashify
        }
        { bbl.page swap$
        }
      if$
      "pages" bibinfo.check
      *
    }
  if$
}
FUNCTION {format.journal.pages}
{ pages duplicate$ empty$ 'pop$
    { swap$ duplicate$ empty$
        { pop$ pop$ format.pages }
        { ", " *
          swap$
          n.dashify
          pages multi.page.check
            'bbl.pages
            'bbl.page
          if$
          swap$
          "pages" bibinfo.check
          * *
        }
      if$
    }
  if$
}
FUNCTION {format.journal.eid}
{ eid "eid" bibinfo.check
  duplicate$ empty$ 'pop$
    { swap$ duplicate$ empty$ 'skip$
      { ", " *
      }
      if$
      swap$ *
    }
  if$
}
FUNCTION {format.vol.num.pages}
{ pubstate field.or.null
  duplicate$ "inpress" =
    { pop$ bbl.inpress }
    { duplicate$ "submitted" =
        { pop$ bbl.submitted }
        { duplicate$ "inpreparation" =
            { pop$ bbl.inpreparation }
            { pop$ "" }
          if$
        }
      if$
    }
  if$
  duplicate$ empty$
    { pop$
      volume field.or.null
      duplicate$ empty$ 'skip$
        { "volume" bibinfo.check
        }
      if$
      number "number" bibinfo.check duplicate$ empty$ 'skip$
        { swap$ duplicate$ empty$
            { "there's a number but no volume in " cite$ * warning$ }
            'skip$
          if$
          swap$
          "(" swap$ * ")" *
        }
      if$ *
      eid empty$
        { format.journal.pages }
        { format.journal.eid }
      if$
    }
    'skip$
  if$
}
FUNCTION {format.chapter.pages}
{ chapter empty$
    'format.pages
    { type empty$
        { bbl.chapter }
        { type "l" change.case$
          "type" bibinfo.check
        }
      if$
      chapter tie.or.space.prefix
      "chapter" bibinfo.check
      * *
      pages empty$
        'skip$
        { ", " * format.pages * }
      if$
    }
  if$
}
FUNCTION {format.journal.series.vol.pages}
{ journal field.or.null
  duplicate$ "OJ" =
    { "journal" bibinfo.check
      emphasize
      series empty$
      volume empty$
      and
      pages empty$
      and
        'skip$
        { " " * }
      if$
      series "series" bibinfo.check
      volume "volume" bibinfo.check *
      duplicate$ empty$
        'skip$
        { pages field.or.null duplicate$ empty$
            'skip$
            { "/" swap$ *
            }
          if$ *
        }
      if$ *
    }
    { duplicate$ "ECR" =
        { "journal" bibinfo.check
          volume "volume" bibinfo.check
          duplicate$ empty$
            'skip$
            { " " swap$ * "--" * * }
          if$
        }
        { "journal" bibinfo.check
          is.online
            { " " * bbl.online * }
            'skip$
          if$
          volume "volume" bibinfo.check
          duplicate$ empty$
            'pop$
            { " " * swap$ * }
          if$
          pages empty$
          eid empty$
          and
            'skip$
            { " " * }
          if$
        }
      if$
    }
  if$
  eid empty$
    { pages "pages" bibinfo.check * }
    { eid "eid" bibinfo.check * }
  if$
}
FUNCTION {format.eprint}
{ eprint duplicate$ empty$
    'skip$
    { "\eprint"
      archive empty$
        'skip$
        { "[" * archive * "]" * }
      if$
      "{" * swap$ * "}" *
    }
  if$
}
FUNCTION {format.url}
{ doi empty$
    { url }
    { "https://doi.org/" doi * }
  if$
  duplicate$ empty$
    { pop$ "" }
    { "\urlprefix\url{" swap$ * "}" * }
  if$
  urlyear empty$
    { urldate empty$
        'skip$
        { " [\urldateprefix{}" * urldate * "]" * }
      if$
    }
    { " [\urldateprefix{}" * urlyear * "]" * }
  if$
}
FUNCTION {format.article.crossref}
{ word.in
  " \cite{" * crossref * "}" *
}
FUNCTION {format.incoll.inproc.crossref}
{ word.in
  " \cite{" * crossref * "}" *
}
FUNCTION {format.book.crossref}
{ volume duplicate$ empty$
    { "empty volume in " cite$ * "'s crossref of " * crossref * warning$
      pop$ word.in
    }
    { bbl.volume
      capitalize
      swap$ tie.or.space.prefix "volume" bibinfo.check * * bbl.of space.word *
    }
  if$
  " \cite{" * crossref * "}" *
}

FUNCTION {article}
{ output.bibitem
  format.authors "author" output.check
  author format.key output
  format.date "year" output.check
  date.block
  format.title "title" output.check
  get.title.addenda
  new.block
  crossref missing$
    { journal "t" change.case$
      "journal" bibinfo.check
      emphasize
      "journal" output.check
      is.online
        { continue.clause
          bbl.online output
        }
        'skip$
      if$
      format.vol.num.pages output
    }
    { format.article.crossref output.nonnull
      format.pages output
    }
  if$
  new.block
  format.note output
  new.block
  format.eprint output
  format.url output
  fin.entry
}
FUNCTION {book}
{ output.bibitem
  author empty$
    { editor empty$
      type$ "reference" =
      or
        { type$ "reference" =
            { format.title "title" output.check }
            { format.btitle "title" output.check }
          if$
          format.date "year" output.check
          year empty$ 'skip$ { date.block } if$
          get.title.addenda
        }
        { format.editors "author and editor" output.check
          format.date "year" output.check
          date.block
          format.btitle "title" output.check
          get.title.addenda
        }
      if$
    }
    { format.authors output.nonnull
      crossref missing$
        { "author and editor" editor either.or.check }
        'skip$
      if$
      format.date "year" output.check
      date.block
      format.btitle "title" output.check
      get.title.addenda
    }
  if$
  format.bvolume output
  format.number.series output
  new.block
  format.edition output
  crossref missing$
    { new.sentence
      format.publisher.address output
    }
    { new.block
      format.book.crossref output.nonnull
    }
  if$
  new.block
  format.note output
  new.block
  format.eprint output
  format.url output
  fin.entry
}
FUNCTION {booklet}
{ output.bibitem
  author empty$
    { format.btitle "title" output.check
      format.date "year" output.check
      year empty$ 'skip$ { date.block } if$
      get.title.addenda
    }
    { format.authors output
      author format.key output
      format.date "year" output.check
      date.block
      format.btitle "title" output.check
      get.title.addenda
    }
  if$
  new.block
  format.booklet.type output
  new.block
  howpublished "howpublished" bibinfo.check output
  new.block
  format.note output
  publisher empty$ 'skip$
    { new.block}
  if$
  format.publisher.address output
  new.block
  format.eprint output
  format.url output
  fin.entry
}
FUNCTION {inbook}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check
      editor format.key output
    }
    { format.authors output.nonnull
      crossref missing$
        { "author and editor" editor either.or.check }
        'skip$
      if$
    }
  if$
  format.date "year" output.check
  date.block
  format.btitle "title" output.check
  get.title.addenda
  new.block
  format.edition output
  crossref missing$
    { format.publisher.address output
      format.bvolume output
      format.chapter.pages "chapter and pages" output.check
      new.block
      format.number.series output
      new.sentence
    }
    { format.chapter.pages "chapter and pages" output.check
      new.block
      format.book.crossref output.nonnull
    }
  if$
  new.block
  format.note output
  new.block
  format.eprint output
  format.url output
  fin.entry
}
FUNCTION {incollection}
{ output.bibitem
  author empty$
    { type$ "inreference" =
        { format.label.booktitle output }
        { format.title "title" output.check }
      if$
      format.date "year" output.check
      year empty$ 'skip$ { date.block } if$
      type$ "inreference" =
        'skip$
        { get.title.addenda }
      if$
    }
    { format.authors output.nonnull
      format.date "year" output.check
      date.block
      format.title "title" output.check
      get.title.addenda
    }
  if$
  new.block
  crossref missing$
    { type$ "inreference" =
      author empty$
      and
        { format.byeditors output }
        { format.in.ed.booktitle "booktitle" output.check }
      if$
      new.block
      format.edition output
      new.block
      type$ "inreference" =
      author empty$
      and
        { format.btitle "title" output.check
          get.title.addenda
          new.block
        }
        'skip$
      if$
      format.publisher.address output
      format.bvolume output
      format.number.series output
      format.chapter.pages output
    }
    { format.incoll.inproc.crossref output.nonnull
      format.chapter.pages output
    }
  if$
  new.block
  format.note output
  new.block
  format.eprint output
  format.url output
  fin.entry
}
FUNCTION {inproceedings}
{ output.bibitem
  format.authors "author" output.check
  author format.key output
  format.date "year" output.check
  date.block
  format.title "title" output.check
  get.title.addenda
  new.block
  crossref missing$
    { format.in.ed.booktitle "booktitle" output.check
      eventyear output
      venue output
      new.sentence
      publisher empty$
        { format.organization.address output }
        { organization "organization" bibinfo.check output
          format.publisher.address output
        }
      if$
      format.bvolume output
      format.number.series output
      format.pages output
    }
    { format.incoll.inproc.crossref output.nonnull
      format.pages output
    }
  if$
  new.block
  format.note output
  new.block
  format.eprint output
  format.url output
  fin.entry
}
FUNCTION {conference} { inproceedings }
FUNCTION {jurisdiction}
{ output.bibitem
  author empty$
    { format.btitle "title" output.check
      continue.clause
      format.case.number output
      number empty$
      casenumber empty$
      and
        'new.sentence
        'continue.clause
      if$
      format.jur.date "year" output.check
      year empty$ 'skip$ { eu.case.check } if$
      get.title.addenda
    }
    { format.authors output
      author format.key output
      format.jur.date "year" output.check
      date.block
      continue.clause
      format.btitle "title" output.check
      get.title.addenda
    }
  if$
  eu.case.check
  format.note output
  note empty$
    'eu.case.check
    'new.block
  if$
  crossref missing$
    { continue.clause
      format.journal.series.vol.pages output
    }
    { format.article.crossref output.nonnull
      format.pages output
    }
  if$
  new.block
  format.eprint output
  format.url output
  fin.entry
}
FUNCTION {uklegislation}
{ output.bibitem
  author empty$
    { format.btitle "title" output.check
      continue.clause
      format.date emphasize "year" output.check
      get.title.addenda
    }
    { format.authors output
      author format.key output
      format.date "year" output.check
      date.block
      continue.clause
      format.btitle "title" output.check
      get.title.addenda
    }
  if$
  entrysubtype field.or.null "secondary" =
    { number "number" bibinfo.check output }
    { series empty$ type empty$ and
        { number empty$
            { format.chapter output }
            { continue.clause
              format.series.number.chapter output
            }
          if$
        }
        { chapter empty$
            { new.block }
            { continue.clause }
          if$
          format.series.number.chapter output
        }
      if$
    }
  if$
  note empty$
    'skip$
    { new.block
      format.note output
      new.block
    }
  if$
  publisher empty$ 'skip$
    { entrysubtype field.or.null "secondary" =
        'skip$
        { new.block }
      if$
      format.publisher.address output
    }
  if$
  new.block
  format.eprint output
  format.url output
  fin.entry
}
FUNCTION {eulegislation}
{ output.bibitem
  author empty$
  { format.title "title" output.check
    continue.clause
    format.eu.date "year" output.check
    get.title.addenda
  }
  { format.authors output
    author format.key output
    format.eu.date "year" output.check
    date.block
    continue.clause
    format.title "title" output.check
    get.title.addenda
  }
  if$
  eu.case.check
  format.note output
  note empty$
    'eu.case.check
    'new.block
  if$
  crossref missing$
    { continue.clause
      format.journal.series.vol.pages output
    }
    { format.article.crossref output.nonnull
      format.pages output
    }
  if$
  new.block
  format.eprint output
  format.url output
  fin.entry
}
FUNCTION {legislation}
{ journal empty$
    'uklegislation
    'eulegislation
  if$
}
FUNCTION {manual}
{ output.bibitem
  author empty$
    { format.btitle "title" output.check
      format.date "year" output.check
      year empty$ 'skip$ { date.block } if$
      get.title.addenda
    }
    { format.authors output
      author format.key output
      format.date "year" output.check
      date.block
      format.btitle "title" output.check
      get.title.addenda
    }
  if$
  format.edition output
  new.block
  format.manual.number output
  new.block
  format.library output
  organization address new.block.checkb
  format.organization.address output
  new.block
  format.note output
  new.block
  format.eprint output
  format.url output
  fin.entry
}
FUNCTION {image}
{ output.bibitem
  author empty$
  { format.btitle "title" output.check
    format.date "year" output.check
    year empty$ 'skip$ { date.block } if$
    get.title.addenda
  }
  { format.authors output
    author format.key output
    format.date "year" output.check
    date.block
    format.btitle "title" output.check
    get.title.addenda
  }
  if$
  format.edition output
  new.block
  format.manual.number output
  new.block
  format.pub.org.lib.address output
  new.block
  format.note output
  new.block
  format.eprint output
  format.url output
  fin.entry
}
FUNCTION {mastersthesis}
{ output.bibitem
  format.authors "author" output.check
  author format.key output
  format.date "year" output.check
  date.block
  format.btitle "title" output.check
  get.title.addenda
  new.block
  bbl.mthesis format.thesis.type output.nonnull
  new.block
  school "school" bibinfo.warn output
  address "address" bibinfo.check output
  new.block
  format.note output
  new.block
  format.eprint output
  format.url output
  fin.entry
}
FUNCTION {misc}
{ output.bibitem
  author empty$
    { format.title "title" output.check
      format.date "year" output.check
      year empty$ 'skip$ { date.block } if$
      get.title.addenda
    }
    { format.authors output
      author format.key output
      format.date "year" output.check
      date.block
      format.title "title" output.check
      get.title.addenda
    }
  if$
  new.block
  howpublished "howpublished" bibinfo.check output
  new.block
  format.note output
  new.block
  format.eprint output
  format.url output
  fin.entry
}
FUNCTION {phdthesis}
{ output.bibitem
  format.authors "author" output.check
  author format.key output
  format.date "year" output.check
  date.block
  format.btitle "title" output.check
  get.title.addenda
  new.block
  bbl.phdthesis format.thesis.type output.nonnull
  new.block
  school "school" bibinfo.warn output
  address "address" bibinfo.check output
  new.block
  format.note output
  new.block
  format.eprint output
  format.url output
  fin.entry
}
FUNCTION {proceedings}
{ output.bibitem
  editor empty$
    { format.btitle "title" output.check
      format.date "year" output.check
      year empty$ 'skip$ { date.block } if$
      get.title.addenda
    }
    { format.editors output
      editor format.key output
      format.date "year" output.check
      date.block
      format.btitle "title" output.check
      get.title.addenda
    }
  if$
  format.bvolume output
  format.number.series output
  new.sentence
  publisher empty$
    { format.organization.address output }
    { organization "organization" bibinfo.check output
      format.publisher.address output
    }
  if$
  new.block
  format.note output
  new.block
  format.eprint output
  format.url output
  fin.entry
}
FUNCTION {techreport}
{ output.bibitem
  author empty$
    { format.btitle "title" output.check
      format.date "year" output.check
      year empty$ 'skip$ { date.block } if$
      get.title.addenda
    }
    { format.authors output
      author format.key output
      format.date "year" output.check
      date.block
      format.btitle "title" output.check
      get.title.addenda
    }
  if$
  new.block
  format.tr.number output
  new.block
  publisher empty$
    { format.institution.address output }
    { institution "institution" bibinfo.check output
      format.publisher.address output
    }
  if$
  new.block
  format.library output
  new.block
  format.note output
  new.block
  format.eprint output
  format.url output
  fin.entry
}
FUNCTION {unpublished}
{ output.bibitem
  author empty$
    { booktitle empty$ 'format.btitle 'format.title if$ "title" output.check
      format.date "year" output.check
      year empty$ 'skip$ { date.block } if$
      get.title.addenda
    }
    { format.authors output
      author format.key output
      format.date "year" output.check
      date.block
      booktitle empty$ 'format.btitle 'format.title if$ "title" output.check
      get.title.addenda
    }
  if$
  new.block
  format.in.ed.booktitle output
  new.block
  howpublished "howpublished" bibinfo.check output
  new.block
  bbl.unpublished output
  new.block
  format.note output
  new.block
  format.eprint output
  format.url output
  fin.entry
}

FUNCTION {letter}
{ journal empty$
    'unpublished
    'article
  if$
}
FUNCTION {collection} { book }
FUNCTION {reference} { book }
FUNCTION {inreference} { incollection }
FUNCTION {audio} { booklet }
FUNCTION {movie} { booklet }
FUNCTION {music} { booklet }
FUNCTION {video} { booklet }
FUNCTION {dataset} { manual }
FUNCTION {electronic} { manual }
FUNCTION {online} { manual }
FUNCTION {patent} { manual }
FUNCTION {software} { manual }
FUNCTION {standard} { manual }
FUNCTION {www} { manual }
FUNCTION {thesis} { phdthesis }
FUNCTION {report} { techreport }
FUNCTION {default.type} { misc }

READ
FUNCTION {sortify}
{ purify$
  "l" change.case$
}
INTEGERS { len }
FUNCTION {chop.word}
{ 's :=
  'len :=
  s #1 len substring$ =
    { s len #1 + global.max$ substring$ }
    's
  if$
}
FUNCTION {format.lab.names}
{'s :=
 "" 't :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr
      "{vv~}{ll}" format.name$
      't :=
      nameptr #1 >
        { nameptr #2 =
          numnames #3 > and
            { "others" 't :=
              #1 'namesleft := }
            'skip$
          if$
          namesleft #1 >
            { ", " * t * }
            { s nameptr "{ll}" format.name$ duplicate$ "others" =
                { 't := }
                { pop$ }
              if$
              t "others" =
                { " " * bbl.etal *
                }
                { bbl.and
                  space.word * t *
                }
              if$
            }
          if$
        }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}
FUNCTION {author.editor.key.label}
{ author empty$
    { editor empty$
        { key empty$
            { title empty$
                { cite$ #1 #3 substring$ }
                { title
                  type$ "book" =
                    { emphasize }
                    'skip$
                  if$
                }
              if$
            }
            'key
          if$
        }
        { editor format.lab.names }
      if$
    }
    { author format.lab.names }
  if$
}
FUNCTION {editor.key.label}
{ editor empty$
    { key empty$
        { title empty$
            { cite$ #1 #3 substring$ }
            { title emphasize }
          if$
        }
        'key
      if$
    }
    { editor format.lab.names }
  if$
}
FUNCTION {author.key.label}
{ author empty$
    { key empty$
        { type$ "inreference" =
          booktitle empty$ not
          and
            { booktitle }
            { title empty$
                { cite$ #1 #3 substring$ }
                { title
                  type$ "article" =
                  type$ "incollection" =
                  or
                  type$ "inproceedings" =
                  or
                    type$ "reference" =
                    author empty$
                    and
                  or
                  type$ "misc" =
                  or
                    'skip$
                    { emphasize }
                  if$
                }
              if$
            }
          if$
        }
        'key
      if$
    }
    { author format.lab.names }
  if$
}
FUNCTION {calc.short.authors}
{ type$ "book" =
  type$ "inbook" =
  or
    'author.editor.key.label
    { type$ "proceedings" =
        'editor.key.label
        'author.key.label
      if$
    }
  if$
  'short.list :=
}
FUNCTION {calc.label}
{ calc.short.authors
  short.list
  "("
  *
  year nodate.check duplicate$ empty$
     { pop$ "" }
     'skip$
  if$
  *
  'label :=
}
FUNCTION {sort.format.names}
{ 's :=
  #1 'nameptr :=
  ""
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr
      "{ll{ }}{  f{ }}{  jj{ }}{ vv{}}"
      format.name$ 't :=
      nameptr #1 >
        { "   "  *
          namesleft #1 =
          t "others" =
          and
            { "zzzzz" 't := }
            'skip$
          if$
          t sortify *
        }
        { t sortify * }
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}
FUNCTION {sort.format.title}
{ 't :=
  "A " #2
    "An " #3
      "The " #4 t chop.word
    chop.word
  chop.word
  sortify
  #1 global.max$ substring$
}
FUNCTION {author.editor.sort}
{ author empty$
    { editor empty$
        { title empty$
            { key empty$
                { "to sort, need author, editor, title, or key in " cite$ * warning$
                  ""
                }
                { key sortify }
              if$
            }
            { title sort.format.title }
          if$
        }
        { editor sort.format.names }
      if$
    }
    { author sort.format.names }
  if$
}
FUNCTION {editor.sort}
{ editor empty$
    { title empty$
        { key empty$
            { "to sort, need editor, title, or key in " cite$ * warning$
              ""
            }
            { key sortify }
          if$
        }
        { title sort.format.title }
      if$
    }
    { editor sort.format.names }
  if$
}
FUNCTION {author.sort}
{ author empty$
    { type$ "inreference" =
      booktitle empty$ not
      and
        { booktitle sort.format.title }
        { title empty$
            { key empty$
                { "to sort, need author, title, or key in " cite$ * warning$
                  ""
                }
                { key sortify }
              if$
            }
            { title sort.format.title }
          if$
        }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {presort}
{ calc.label
  label sortify
  "    "
  *
  type$ "book" =
  type$ "inbook" =
  or
    'author.editor.sort
    { type$ "proceedings" =
        'editor.sort
        'author.sort
      if$
    }
  if$
  #1 entry.max$ substring$
  'sort.label :=
  sort.label
  *
  "    "
  *
  title field.or.null
  sort.format.title
  *
  #1 entry.max$ substring$
  'sort.key$ :=
}

ITERATE {presort}
SORT
STRINGS { last.label next.extra }
INTEGERS { last.extra.num last.extra.num.extended last.extra.num.blank number.label }
FUNCTION {initialize.extra.label.stuff}
{ #0 int.to.chr$ 'last.label :=
  "" 'next.extra :=
  #0 'last.extra.num :=
  "a" chr.to.int$ #1 - 'last.extra.num.blank :=
  last.extra.num.blank 'last.extra.num.extended :=
  #0 'number.label :=
}
FUNCTION {forward.pass}
{ last.label label =
    { last.extra.num #1 + 'last.extra.num :=
      last.extra.num "z" chr.to.int$ >
       { "a" chr.to.int$ 'last.extra.num :=
         last.extra.num.extended #1 + 'last.extra.num.extended :=
       }
       'skip$
      if$
      last.extra.num.extended last.extra.num.blank >
        { last.extra.num.extended int.to.chr$
          last.extra.num int.to.chr$
          * 'extra.label := }
        { last.extra.num int.to.chr$ 'extra.label := }
      if$
    }
    { "a" chr.to.int$ 'last.extra.num :=
      "" 'extra.label :=
      label 'last.label :=
    }
  if$
  number.label #1 + 'number.label :=
}
FUNCTION {reverse.pass}
{ next.extra "b" =
    { "a" 'extra.label := }
    'skip$
  if$
  extra.label 'next.extra :=
  extra.label
  duplicate$ empty$
    'skip$
    { "{\natexlab{" swap$ * "}}" * }
  if$
  'extra.label :=
  label extra.label * 'label :=
}
EXECUTE {initialize.extra.label.stuff}
ITERATE {forward.pass}
REVERSE {reverse.pass}
FUNCTION {bib.sort.order}
{ sort.label
  "    "
  *
  year field.or.null sortify
  *
  "    "
  *
  title field.or.null
  sort.format.title
  *
  #1 entry.max$ substring$
  'sort.key$ :=
}
ITERATE {bib.sort.order}
SORT
FUNCTION {begin.bib}
{ preamble$ empty$
    'skip$
    { preamble$ write$ newline$ }
  if$
  "\begin{thebibliography}{" number.label int.to.str$ * "}" *
  write$ newline$
  "\providecommand{\natexlab}[1]{#1}"
  write$ newline$
  "\providecommand{\url}[1]{\texttt{#1}}"
  write$ newline$
  "\providecommand{\urlprefix}{Available from: }"
  write$ newline$
  "\providecommand{\urldateprefix}{Accessed }"
  write$ newline$
  "\providecommand{\selectlanguage}[1]{\relax}"
  write$ newline$
  "\providecommand{\bibinfo}[2]{#2}"
  write$ newline$
  "\providecommand{\eprint}[2][]{\url{#2}}"
  write$ newline$
}
EXECUTE {begin.bib}
EXECUTE {init.state.consts}
ITERATE {call.type$}
FUNCTION {end.bib}
{ newline$
  "\end{thebibliography}" write$ newline$
}
EXECUTE {end.bib}
%% 
%% Copyright (C) 2016-2021 by University of Bath
%%
%% End of file `bathx.bst'.
