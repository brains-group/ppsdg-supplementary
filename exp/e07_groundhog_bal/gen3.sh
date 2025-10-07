for d in BC:4074 BM:10578 CC:13272 CR:600 GM:20052; do
	n=${d#*:}
	for file in ${d%:*}_*; do
		gile=${file/_bal/_bal2}
		for j in 0 {10..19}; do
			for io in in:true out:false; do
				shf=config/${gile%.yaml}-shadow$j-${io%:*}.yaml
				[ -f "$shf" ] || {
					cat "$file"
					printf '%s\n' 'canary_index: 101' "include_canary: ${io#*:}" "reference_size: $((n/2))" "reference_seed: $((j/10+21))" "sample_size: $((n/5))" "sample_seed: $((j+30))"
				} > "$shf"
			done
		done

		ghf=config/${gile%.yaml}-groundhog.yaml
		[ -f "$ghf" ] || (
		wd=$PWD; cd ~/ppsdg;
		sed -n /dataset:/p "$wd/$file"
		set -- `~/miniconda3/envs/ppsdg/bin/train-gen -n "$wd/config/${gile%.yaml}-shadow"{0,{10..19}}-{out,in}.yaml`
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
