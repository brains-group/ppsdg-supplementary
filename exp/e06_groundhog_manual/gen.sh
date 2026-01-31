for d in AD:48842 BC:10000 BM:45211 CC:30000 CR:1000 GM:150000; do
	n=${d#*:}
	for file in ${d%:*}_*; do
		for j in 0 {10..19}; do
			for io in in:true out:false; do
				shf=config/${file%.yaml}-shadow$j-${io%:*}.yaml
				[ -f "$shf" ] || {
					cat "$file"
					printf '%s\n' 'canary_index: 100' "include_canary: ${io#*:}" "reference_size: $((n/2))" "reference_seed: $((j/10+1))" "sample_size: $((n/5))" "sample_seed: $((j+10))"
				} > "$shf"
			done
		done

		ghf=config/${file%.yaml}-groundhog.yaml
		[ -f "$ghf" ] || (
		wd=$PWD; cd ~/ppsdg;
		sed -n /dataset:/p "$wd/$file"
		set -- `~/miniconda3/envs/ppsdg/bin/train-gen -n "$wd/config/${file%.yaml}-shadow"{0,{10..19}}-{out,in}.yaml`
		bname=model.joblib
		case $file in
		*) #*tabdiff*)
			bname=default-44136fa3.csv ;;
		esac
		echo train_models:
		printf "  - %s/$bname\n" "$@" | grep w1.-out | sed '1s/^./-/'
		printf "  - %s/$bname\n" "$@" | grep w1.-in | sed '1s/^./-/'
		echo test_models:
		printf "- - %s/$bname\n" "$@" | grep w0-out
		printf "- - %s/$bname\n" "$@" | grep w0-in) > "$ghf" &
	done
done
wait
