"""Record maintained, research, historical and test definitions without imports."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import subprocess


def category(path):
    parts=path.parts
    if parts[0]=='openspliceai':
        if {'scripts','test'}.intersection(parts) or path.name in ('get_anno.py','gff_to_tsv.py','temperature_scaling_site_only.py'):
            return 'historical'
        return 'maintained_package'
    if parts[0]=='validation':
        return 'research_campaign'
    if parts[0]=='tests':
        return 'verification'
    if parts[0]=='experiments':
        return 'historical'
    return 'tooling_examples'


def inventory(root):
    output=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','--','*.py'],cwd=root,text=True)
    records=[]
    for name in sorted(set(output.splitlines())):
        path=Path(name)
        source=(root/path).read_text()
        row={'path':name,'category':category(path),'lines':len(source.splitlines()),'definitions':[]}
        try:
            tree=ast.parse(source)
        except SyntaxError as error:
            row['parse_error']=f'{error.msg} at line {error.lineno}'
            records.append(row)
            continue
        def walk(nodes,prefix='',inside_function=False,private_parent=False):
            for node in nodes:
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                    qualified=prefix+node.name
                    row['definitions'].append({'name':qualified,'line':node.lineno,'end_line':node.end_lineno,
                        'kind':'class' if isinstance(node,ast.ClassDef) else 'function',
                        'documented':bool(ast.get_docstring(node)),
                        'public':not inside_function and not private_parent and not node.name.startswith('_')})
                    walk(node.body,qualified+'.',inside_function or not isinstance(node,ast.ClassDef),
                         private_parent or node.name.startswith('_'))
        walk(tree.body)
        records.append(row)
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    records=inventory(args.root)
    totals=Counter(record['category'] for record in records)
    summary={'files':dict(totals),'lines':{name:sum(row['lines'] for row in records if row['category']==name) for name in totals},
             'definitions':{name:sum(len(row['definitions']) for row in records if row['category']==name) for name in totals}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'summary':summary,'files':records},indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
