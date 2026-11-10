"""Prepare the ten benchmark classes using official cleaned DomainNet lists."""
from pathlib import Path, PurePosixPath
import argparse, shutil, subprocess, zipfile, json
BASE='https://csr.bu.edu/ftp/visda/2019/multi-source/'
DOMAINS=('clipart','infograph','painting','quickdraw','real','sketch')
CLASSES=('bird','cat','dog','elephant','horse','airplane','bicycle','car','motorbike','truck')

def download(url,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file() and path.stat().st_size:return
    part=path.with_name(path.name+'.part')
    print('Downloading:',url,flush=True)
    subprocess.run(['curl','--fail','--location','--retry','5','--connect-timeout','30',
                    '--continue-at','-','--output',str(part),url],check=True)
    part.replace(path)

def selected_paths(text,domain):
    found=[];counts={c:0 for c in CLASSES}
    for line in text.splitlines():
        if not line.strip():continue
        rel,label=line.rsplit(maxsplit=1);int(label)
        p=PurePosixPath(rel)
        if p.is_absolute() or '..' in p.parts or len(p.parts)<3 or p.parts[0]!=domain:
            raise ValueError('Unexpected split path: '+rel)
        if p.parts[-2] in counts:
            found.append(rel);counts[p.parts[-2]]+=1
    if any(v==0 for v in counts.values()):raise ValueError(f'{domain}: missing classes in split: {counts}')
    if len(found)!=len(set(found)):raise ValueError('Duplicate split entries')
    return found

def extract_selected(archive,root,paths):
    root=Path(root).resolve()
    with zipfile.ZipFile(archive) as z:
        for rel in sorted(paths):
            info=z.getinfo(rel);dest=root/rel
            if not dest.resolve().is_relative_to(root):raise ValueError('Unsafe extraction path')
            # Verify selected existing files against ZIP CRC before reusing them.
            import zlib
            if dest.is_file() and dest.stat().st_size==info.file_size:
                with dest.open('rb') as f:
                    crc=0
                    for chunk in iter(lambda:f.read(1024*1024),b''):crc=zlib.crc32(chunk,crc)
                if crc==info.CRC:continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            tmp=dest.with_name(dest.name+'.extracting')
            with z.open(info) as source,tmp.open('wb') as out:shutil.copyfileobj(source,out)
            tmp.replace(dest)

def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    report={}
    for domain in DOMAINS:
        paths={}
        for split in ('train','test'):
            path=root/f'{domain}_{split}.txt'
            download(BASE+f'domainnet/txt/{path.name}',path)
            paths[split]=selected_paths(path.read_text(),domain)
        if set(paths['train']) & set(paths['test']):raise ValueError('Train/test image overlap')
        archive=root/'archives'/f'{domain}.zip'
        prefix='groundtruth/' if domain in ('clipart','painting') else ''
        download(BASE+prefix+domain+'.zip',archive)
        extract_selected(archive,root,set(paths['train'])|set(paths['test']))
        report[domain]={s:len(v) for s,v in paths.items()}
        print(domain,report[domain],flush=True)
    (root/'preparation_report.json').write_text(json.dumps(report,indent=2))
    print('Preparation complete:',root,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    prepare(p.parse_args().root)
