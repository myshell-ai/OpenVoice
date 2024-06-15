import math
import subprocess


print('''Contributors
============

All contributors (by number of commits):
''')


email_map = {

    # Maintainers.
    'git@mikeboers.com': 'github@mikeboers.com',
    'mboers@keypics.com': 'github@mikeboers.com',
    'mikeb@loftysky.com': 'github@mikeboers.com',
    'mikeb@markmedia.co': 'github@mikeboers.com',
    'westernx@mikeboers.com': 'github@mikeboers.com',

    # Junk.
    'mark@mark-VirtualBox.(none)': None,

    # Aliases.
    'a.davoudi@aut.ac.ir': 'davoudialireza@gmail.com',
    'tcaswell@bnl.gov': 'tcaswell@gmail.com',
    'xxr3376@gmail.com': 'xxr@megvii.com',
    'dallan@pha.jhu.edu': 'daniel.b.allan@gmail.com',

}

name_map = {
    'caspervdw@gmail.com': 'Casper van der Wel',
    'daniel.b.allan@gmail.com': 'Dan Allan',
    'mgoacolou@cls.fr': 'Manuel Goacolou',
    'mindmark@gmail.com': 'Mark Reid',
    'moritzkassner@gmail.com': 'Moritz Kassner',
    'vidartf@gmail.com': 'Vidar Tonaas Fauske',
    'xxr@megvii.com': 'Xinran Xu',
}

github_map = {
    'billy.shambrook@gmail.com': 'billyshambrook',
    'daniel.b.allan@gmail.com': 'danielballan',
    'davoudialireza@gmail.com': 'adavoudi',
    'github@mikeboers.com': 'mikeboers',
    'jeremy.laine@m4x.org': 'jlaine',
    'kalle.litterfeldt@gmail.com': 'litterfeldt',
    'mindmark@gmail.com': 'markreidvfx',
    'moritzkassner@gmail.com': 'mkassner',
    'rush@logic.cz': 'radek-senfeld',
    'self@brendanlong.com': 'brendanlong',
    'tcaswell@gmail.com': 'tacaswell',
    'ulrik.mikaelsson@magine.com': 'rawler',
    'vidartf@gmail.com': 'vidartf',
    'willpatera@gmail.com': 'willpatera',
    'xxr@megvii.com': 'xxr3376',
}


email_count = {}
for line in subprocess.check_output(['git', 'log', '--format=%aN,%aE']).decode().splitlines():
    name, email = line.strip().rsplit(',', 1)

    email = email_map.get(email, email)
    if not email:
        continue

    names = name_map.setdefault(email, set())
    if isinstance(names, set):
        names.add(name)

    email_count[email] = email_count.get(email, 0) + 1


last = None
block_i = 0
for email, count in sorted(email_count.items(), key=lambda x: (-x[1], x[0])):

    # This is the natural log, because of course it should be. ;)
    order = int(math.log(count))
    if last and last != order:
        block_i += 1
        print()
    last = order

    names = name_map[email]
    if isinstance(names, set):
        name = ', '.join(sorted(names))
    else:
        name = names

    github = github_map.get(email)

    # The '-' vs '*' is so that Sphinx treats them as different lists, and
    # introduces a gap bettween them.
    if github:
        print('%s %s <%s>; `@%s <https://github.com/%s>`_' % ('-*'[block_i % 2], name, email, github, github))
    else:
        print('%s %s <%s>'      % ('-*'[block_i % 2], name, email,       ))
